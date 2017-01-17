package edu.stanford.nlp.hcoref.train;

import java.io.File;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import hr.irb.fastRandomForest.FastRandomForest;
import weka.core.Instances;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.hcoref.CorefDocMaker;
import edu.stanford.nlp.hcoref.CorefProperties;
import edu.stanford.nlp.hcoref.data.Dictionaries;
import edu.stanford.nlp.hcoref.data.Document;
import edu.stanford.nlp.hcoref.data.Mention;
import edu.stanford.nlp.hcoref.md.MentionDetectionClassifier;
import edu.stanford.nlp.hcoref.rf.RandomForest;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Generics;

public class MentionDetectionClassifierFactory {
  public static RVFDataset<Boolean, String> extractTrainData(Properties props) throws Exception {
    String timeStamp = Calendar.getInstance().getTime().toString().replaceAll("\\s", "-").replaceAll(":", "-");
    Dictionaries dict = new Dictionaries(props);
    CorefDocMaker docMaker = new CorefDocMaker(props, dict);
//    props.put(CorefProperties.CONLL2011_PROP, props.getProperty("PATH_TRAINDATA"));
//    MentionExtractor mentionExtractor = new CoNLLMentionExtractor(dict, props, null, null);
//    mentionExtractor.resetDocs();
    RVFDataset<Boolean, String> trainData = new RVFDataset<Boolean, String>();
    
    // run processes
    while (true) {
      Document document = docMaker.nextDoc();
      if (document == null) break;
      document.extractGoldCorefClusters();
      
      // extract named entity spans
      Set<String> neStrings = Generics.newHashSet();
      for(int i=0 ; i<document.predictedMentions.size() ; i++) {
        for(Mention m : document.predictedMentions.get(i)) {
          String ne = m.headWord.ner();
          if(ne.equals("O")) continue;
          for(CoreLabel cl : m.originalSpan) {
            if(!cl.ner().equals(ne)) continue;
          }
          neStrings.add(m.lowercaseNormalizedSpanString());
        }
      }
      
      for(int i=0 ; i<document.predictedMentions.size() ; i++) {
        List<Mention> golds = document.goldMentions.get(i);
        List<Mention> predicts = document.predictedMentions.get(i);
        
        Map<Integer, Set<Mention>> headPositions = new HashMap<>();
        for(Mention p : predicts) {
          if (!headPositions.containsKey(p.headIndex)) headPositions.put(p.headIndex, new HashSet<>());
          headPositions.get(p.headIndex).add(p);
        }
        
        for(Mention g : golds) {
          if (!headPositions.containsKey(g.headIndex)) headPositions.put(g.headIndex, new HashSet<>());
          Set<Mention> shares = headPositions.get(g.headIndex);
          if(shares.size() > 1) {
            for(Mention p : shares) {
              boolean isMention = (g.startIndex == p.startIndex && g.endIndex == p.endIndex);
              trainData.add(new RVFDatum<Boolean, String>(MentionDetectionClassifier.extractFeatures(p, shares, neStrings, dict, props), isMention));
            }
          }
        }
      }
    }
    return trainData;
  }
  public static MentionDetectionClassifier train(RVFDataset<Boolean, String> traindata, Properties props) throws Exception {
    traindata.applyFeatureCountThreshold(CorefProperties.getFeatureCountThreshold(props, "md"));
    System.err.println("converting data.....");
    Instances trainSet = DataConverter.convertGeneralDatasetToInstances(traindata, true);
    
    FastRandomForest fastrf = new FastRandomForest();
    fastrf.setOptions(new String[]{
        "-I", Integer.toString(CorefProperties.getNumTrees(props, "md")), 
        "-threads", Integer.toString(0),
        "-S", Integer.toString(CorefProperties.getSeed(props)),
        "-K", Integer.toString(CorefProperties.getNumFeatures(props, "md")),
        "-depth", Integer.toString(CorefProperties.getTreeDepth(props, "md")),
    });
    fastrf.setComputeImportances(CorefProperties.calculateFeatureImportance(props));
    
    Date startTime = new Date();
    System.err.println(" classifier training... (training set size: "+trainSet.numInstances()+"), # of features: "+trainSet.numAttributes());
    System.err.printf("\t\ttrain start time: %s\n", startTime);
    fastrf.buildClassifier(trainSet);
    System.err.printf("\t\telapsed time: %.3f seconds\n", (((new Date()).getTime() - startTime.getTime()) / 1000F));
    
    if(fastrf.getComputeImportances()) {
      CorefTrainer.printRFFeatureImportance(fastrf);
    }
    
    RandomForest rf = RandomForestConverter.convert(fastrf);
    
    return new MentionDetectionClassifier(rf);
  }
  public static void trainMentionSpanClassifier(Properties props) throws Exception {
    props.put(CorefProperties.PATH_INPUT_PROP, CorefProperties.getPathTrainData(props));
    System.err.println(props);
    File model = new File(CorefProperties.getPathModel(props, "md"));
    if(model.exists()) {
      System.err.println("MD model exists!");
      return;
    }
    model.getParentFile().mkdirs();
    props.setProperty(CorefProperties.MD_TRAIN_PROP, "true");
    
    RVFDataset<Boolean, String> trainData = extractTrainData(props);
    System.err.println("trainData size: "+trainData.size());
    
    if(CorefProperties.doPMIFeatureSelection(props, "md")) {
      Counter<String> pmi = MutualInformationFeatureSelection.featureSelectionByMutualInfo(trainData);
      double pmiThres = CorefProperties.getPMIThres(props, "md");
      Counters.retainAbove(pmi, pmiThres);
      trainData = MutualInformationFeatureSelection.retainNumericOrSelected(trainData, pmi.keySet());
    }
    
    
    MentionDetectionClassifier mdClassifier = train(trainData, props);
    IOUtils.writeObjectToFile(mdClassifier, model);

    props.remove(CorefProperties.MD_TRAIN_PROP);
  }
}
