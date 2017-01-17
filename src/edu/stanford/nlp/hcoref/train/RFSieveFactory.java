package edu.stanford.nlp.hcoref.train;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import hr.irb.fastRandomForest.FastRandomForest;
import weka.core.Instances;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.hcoref.CorefProperties;
import edu.stanford.nlp.hcoref.CorefSystem;
import edu.stanford.nlp.hcoref.data.Document;
import edu.stanford.nlp.hcoref.data.Mention;
import edu.stanford.nlp.hcoref.rf.RandomForest;
import edu.stanford.nlp.hcoref.sieve.RFSieve;
import edu.stanford.nlp.hcoref.sieve.Sieve;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.concurrent.MulticoreWrapper;
import edu.stanford.nlp.util.concurrent.ThreadsafeProcessor;
import edu.stanford.nlp.util.logging.Redwood;

/** train one sieve or load a list of sieves */
public class RFSieveFactory {
  
  // extract traindata
  public static RVFDataset<Boolean, String> extractTrainData(Properties props, String sievename) throws Exception {
    
    CorefSystem cs = new CorefSystem(props);
    cs.docMaker.resetDocs();
    
    // extract dataset
    RVFDataset<Boolean, String> traindata = new RVFDataset<Boolean, String>();
    
    int nThreads = CorefProperties.getThreadCounts(props);
    System.err.println("Number of Threads: " + nThreads);
    MulticoreWrapper<List<Object>, RVFDataset<Boolean, String>> wrapper = 
        new MulticoreWrapper<List<Object>, RVFDataset<Boolean, String>>(
        nThreads, new ThreadsafeProcessor<List<Object>, RVFDataset<Boolean, String>>() {
          @Override
          public RVFDataset<Boolean, String> process(List<Object> input) {
            try {
              RVFDataset<Boolean, String> traindataOneDoc = new RVFDataset<Boolean, String>();
              RVFDataset<Boolean, String> trainPositiveData = new RVFDataset<Boolean, String>();
              RVFDataset<Boolean, String> trainNegativeData = new RVFDataset<Boolean, String>();

              Document document = (Document) input.get(0);
              CorefSystem cs = (CorefSystem) input.get(1);
              Properties props = cs.props;
              
              String year = document.docInfo.get("DOC_FILE").contains("conll-2011")? "2011" : "2012"; 
              
              document.extractGoldCorefClusters();
              
              for(Sieve sieve : cs.sieves){
                sieve.resolveMention(document, cs.dictionaries, props);
              }
              
              for(int sentIdx=0 ; sentIdx < document.predictedMentions.size() ; sentIdx++) {
                List<Mention> mentionsInSent = document.predictedMentions.get(sentIdx);
                
                for(int mIdx = 0 ; mIdx < mentionsInSent.size() ; mIdx++) {
                  Mention m = mentionsInSent.get(mIdx);
//                  CorefCluster mC = document.corefClusters.get(m.corefClusterID);
//                  if(mC.getRepresentativeMention() != m) {
//                    continue;
//                  }
                  
                  boolean corefFound = false;
                  int mentionDist = 0;
                  for(int sentDist=0 ; sentDist <= Math.min(CorefProperties.getMaxSentDistForSieve(props, sievename), sentIdx) ; sentDist++) {
                    List<Mention> candidates = Sieve.getOrderedAntecedents(m, sentIdx-sentDist, mIdx, document.predictedMentions, cs.dictionaries); 
//                      List<Mention> candidates = document.predictedOrderedMentionsBySentence.get(sentIdx-sentDist);
                    
                    for(Mention candidate : candidates) {
                      if(candidate == m) continue;
                      if(!CorefProperties.getMentionType(props, sievename).contains(m.mentionType) || !CorefProperties.getAntecedentType(props, sievename).contains(candidate.mentionType)) continue;
                      if(sentDist==0 && m.appearEarlierThan(candidate)) continue;   // ignore cataphora
                      mentionDist++;
                      
                      RVFDatum<Boolean, String> datum = RFSieve.extractDatum(m, candidate, document, mentionDist, cs.dictionaries, props, sievename);

//                      traindataOneDoc.add(datum);
                      if(datum.label()) {
                        corefFound = true;    // extract until the same sentence of the first coref antecedent
                        Sieve.merge(document, m.mentionID, candidate.mentionID);
                        trainPositiveData.add(datum);
                      } else {
                        trainNegativeData.add(datum);
                      }
                    }
                    if(corefFound) break;   // extract until the same sentence of the first coref antecedent
                  }
                }
              }
              traindataOneDoc.addAll(trainPositiveData);
//              traindataOneDoc.addAll(trainNegativeData.sampleDataset(0, options.downsamplingRate, false));
              traindataOneDoc.addAll(trainNegativeData);

              return traindataOneDoc;
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
          }

          @Override
          public ThreadsafeProcessor<List<Object>, RVFDataset<Boolean, String>> newInstance() {
            return this;
          }
        });

    // run processes
    int i = 0;
    while (true) {
      Document document = cs.docMaker.nextDoc();
      if (document == null) break;
      List<Object> input = new ArrayList<Object>();
      input.add(document);
      input.add(cs);
      wrapper.put(input);
      while (wrapper.peek()) {
        RVFDataset<Boolean, String> traindataOneDoc = wrapper.poll();
        traindata.addAll(traindataOneDoc);
        if ((i++) % 10 == 0) System.err.println(i + " document(s) processed");
      }
    }

    // Finished reading the input. Wait for jobs to finish
    wrapper.join();
    while (wrapper.peek()) {
      RVFDataset<Boolean, String> traindataOneDoc = wrapper.poll();
      traindata.addAll(traindataOneDoc);
      if ((i++) % 10 == 0) System.err.println(i + " document(s) processed");
    }

    
    System.err.println("traindata size: "+traindata.size());
    System.err.println("traindata feature size: "+traindata.numFeatures());
    
    return traindata;
  }

  // train a sieve
  public static RFSieve boostTrain(RVFDataset<Boolean, String> traindata, String pathModel, Properties props, String sievename) throws Exception {
    System.err.println("TRAINING DATA SIZE: "+traindata.size());
    traindata.applyFeatureCountThreshold(CorefProperties.getFeatureCountThreshold(props, sievename));

    RandomForest rf = null;
    RFSieve sieve = new RFSieve(rf, props, sievename);
    
    int MAX_ITER = (CorefProperties.getDownsamplingRate(props, sievename)==1)? 1 : 2;    // if we don't downsample, 1 iteration is enough
    for(int iter=0 ; iter<=MAX_ITER ; iter++) {
      if((new File(pathModel.replace("model.ser", "model"+iter+".ser"))).exists()) {
        System.err.println("read serialized previous model: "+pathModel.replace("model.ser", "model"+iter+".ser"));
        sieve = IOUtils.readObjectFromFile(pathModel.replace("model.ser", "model"+iter+".ser"));
        continue;
      }
      
      System.err.println("TRAINING... iter: "+iter);

      RVFDataset<Boolean, String> downsampled = downsampledTraindata(traindata, sieve, props);
      
      if(iter==MAX_ITER) break;   // to see the trained model performance on training data
      
      rf = trainRF(downsampled, props, sievename);
      sieve = new RFSieve(rf, props, sievename);
      
      // store intermediate models
      if(iter<MAX_ITER-1) {
        IOUtils.writeObjectToFile(sieve, pathModel.replace("model.ser", "model"+iter+".ser"));
      }
    }
    
    return sieve;
  }
  
  private static RVFDataset<Boolean, String> downsampledTraindata(RVFDataset<Boolean, String> traindata, RFSieve sieve, Properties props) throws Exception {
    if(CorefProperties.getDownsamplingRate(props, sieve.sievename)==1) return traindata;   // no downsampling
    
    RandomForest rf = sieve.rf;
    
    RVFDataset<Boolean, String> downsampled = new RVFDataset<Boolean, String>();
    
    Map<String, RVFDatum<Boolean, String>> positiveMap = Generics.newHashMap();
    Counter<String> positiveScore = new ClassicCounter<String>();
    
    Map<String, RVFDatum<Boolean, String>> negativeMap = Generics.newHashMap();
    Counter<String> negativeScore = new ClassicCounter<String>();
    
    int nThreads = CorefProperties.getThreadCounts(props);
    System.err.println("Number of Threads: " + nThreads);
    
    // input: (traindataPerThread, rf, instancesForAttrInfo, options), output: (positivesPerThread, negativeHardPerThread, negativeEasyPerThread)
    MulticoreWrapper<List<Object>, List<Object>> wrapper = 
        new MulticoreWrapper<List<Object>, List<Object>>(
        nThreads, new ThreadsafeProcessor<List<Object>, List<Object>>() {
          @Override
          public List<Object> process(List<Object> input) {
            try {
              Map<String, RVFDatum<Boolean, String>> positiveMapPerThread = Generics.newHashMap();
              Counter<String> positiveScorePerThread = new ClassicCounter<String>();
              
              Map<String, RVFDatum<Boolean, String>> negativeMapPerThread = Generics.newHashMap();
              Counter<String> negativeScorePerThread = new ClassicCounter<String>();
              
              RVFDataset<Boolean, String> traindataPerThread = (RVFDataset<Boolean, String>) input.get(0);
              RandomForest rf = (RandomForest) input.get(1);
              int threadIdx = (int) input.get(2);
              
              int id=0;
              for(RVFDatum<Boolean, String> datum : traindataPerThread) {
                id++;
                String idKey = threadIdx+"-"+id;
                
                if(rf==null) {
                  if(datum.label()) {   // positive datum: coreferent
                    positiveMapPerThread.put(idKey, datum);
                    positiveScorePerThread.incrementCount(idKey, Math.random());
                  } else {    // negative datum: not coreferent
                    negativeMapPerThread.put(idKey, datum);
                    negativeScorePerThread.incrementCount(idKey, Math.random());  // random sampling at first
                  }
                } else {
                  double probTrue = rf.probabilityOfTrue(datum);
                  
                  if(datum.label()) {   // positive datum: coreferent
                    positiveMapPerThread.put(idKey, datum);
                    positiveScorePerThread.incrementCount(idKey, probTrue);
                  } else {    // negative datum: not coreferent
                    negativeMapPerThread.put(idKey, datum);
                    negativeScorePerThread.incrementCount(idKey, probTrue);
                  }
                }
              }
              List<Object> output = Generics.newArrayList();
              
              output.add(positiveMapPerThread);
              output.add(positiveScorePerThread);
              output.add(negativeMapPerThread);
              output.add(negativeScorePerThread);
              
              return output;
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
          }

          @Override
          public ThreadsafeProcessor<List<Object>, List<Object>> newInstance() {
            return this;
          }
        });

    // run processes
    int splitSize = traindata.size()/nThreads;
    for (int i=0 ; i<nThreads ; i++) {
      
      int start = i*splitSize;
      int end = (i==nThreads-1)? traindata.size() : (i+1)*splitSize;
      RVFDataset<Boolean, String> traindataPerThread = (RVFDataset<Boolean, String>) traindata.split(start, end).second();
      
      List<Object> input = new ArrayList<Object>();
      input.add(traindataPerThread);
      input.add(rf);
      input.add(i);
      wrapper.put(input);
      while (wrapper.peek()) {
        List<Object> output = wrapper.poll();
        positiveMap.putAll((Map<String, RVFDatum<Boolean, String>>) output.get(0));
        positiveScore.addAll((Counter<String>) output.get(1));
        negativeMap.putAll((Map<String, RVFDatum<Boolean, String>>) output.get(2));
        negativeScore.addAll((Counter<String>) output.get(3));
      }
    }

    // Finished reading the input. Wait for jobs to finish
    wrapper.join();
    while (wrapper.peek()) {
      List<Object> output = wrapper.poll();
      positiveMap.putAll((Map<String, RVFDatum<Boolean, String>>) output.get(0));
      positiveScore.addAll((Counter<String>) output.get(1));
      negativeMap.putAll((Map<String, RVFDatum<Boolean, String>>) output.get(2));
      negativeScore.addAll((Counter<String>) output.get(3));
    }

    System.err.println("original size: "+traindata.size());
    System.err.println("positives size: "+positiveMap.size());
    System.err.println("negatives size: "+negativeMap.size());
    
    // set thresMerge by best F1 score on trainset
    double maxF1Thres = -1;
    double maxF1 = -1;
    DecimalFormat df = new DecimalFormat("#.####");
    for(double thres = 0.1 ; thres < 1 ; thres=thres+0.01) {
      int posCorrect = Counters.keysAbove(positiveScore, thres).size();
      int posIncorrect = positiveMap.size() - posCorrect;
      int negIncorrect = Counters.keysAbove(negativeScore, thres).size();
      int negCorrect = negativeMap.size() - negIncorrect;
      
      double p = 1.0 * posCorrect / (posCorrect + negIncorrect);
      double r = 1.0 * posCorrect / (posCorrect + posIncorrect);
      double f1 = 2*p*r/(p+r);
      
      Redwood.log("debug-mergethres", "thres: "+thres +", f1: "+df.format(f1)+", r: "+df.format(r)+", p: "+df.format(p)+", posCorrect: "+posCorrect+", posIncorrect: "+posIncorrect+", negCorrect: "+negCorrect+", negIncorrect: "+negIncorrect);

      if(maxF1 < f1) {
        maxF1 = f1;
        maxF1Thres = thres;
      }
    }
    System.err.println("maxF1: "+maxF1);
    System.err.println("maxF1Thres: "+maxF1Thres);
    CorefProperties.setMergeThreshold(props, sieve.sievename, maxF1Thres);
    
    int retainTopSize = (int) Math.floor(negativeMap.size()*CorefProperties.getDownsamplingRate(props, sieve.sievename));
    System.err.println("retainTopSize: "+retainTopSize);
    Counters.retainTop(negativeScore, retainTopSize);
    negativeMap.keySet().retainAll(negativeScore.keySet());
    
    downsampled.addAll(positiveMap.values());
    downsampled.addAll(negativeMap.values());
    
    System.err.println("downsampled size: "+downsampled.size());
    
    return downsampled;
  }

  public static RandomForest trainRF(RVFDataset<Boolean, String> traindata, Properties props, String sievename) throws Exception {
    Instances trainSet = DataConverter.convertGeneralDatasetToInstances(traindata, true);
    FastRandomForest fastrf = trainRF(trainSet, props, sievename);
    trainSet.delete();
    return RandomForestConverter.convert(fastrf);
  }
  public static FastRandomForest trainRF(Instances trainSet, Properties props, String sievename) throws Exception {
    FastRandomForest rf = new FastRandomForest();
    rf.setOptions(new String[]{
        "-I", Integer.toString(CorefProperties.getNumTrees(props, sievename)), 
        "-threads", Integer.toString(0),
        "-S", Integer.toString(CorefProperties.getSeed(props)),
        "-K", Integer.toString(CorefProperties.getNumFeatures(props, sievename)),
        "-depth", Integer.toString(CorefProperties.getTreeDepth(props, sievename)),
    });
    rf.setComputeImportances(CorefProperties.calculateFeatureImportance(props));
    
    Date startTime = new Date();
    System.err.println(" classifier training... (training set size: "+trainSet.numInstances()+"), # of features: "+trainSet.numAttributes());
    System.err.printf("\t\ttrain start time: %s\n", startTime);
    rf.buildClassifier(trainSet);
    System.err.printf("\t\telapsed time: %.3f seconds\n", (((new Date()).getTime() - startTime.getTime()) / 1000F));
    
    if(rf.getComputeImportances()) {
      CorefTrainer.printRFFeatureImportance(rf);
    }
    
    return rf;
  }
  
  public static void trainSieve(Properties props, String sievename) throws Exception {
    String timeStamp = Calendar.getInstance().getTime().toString().replaceAll("\\s", "-").replaceAll(":", "-");
    System.err.println(timeStamp);
    System.err.println(props.toString());
    
    // make directory if not exists
    File dir = new File(CorefProperties.getPathSerialized(props));
    dir.mkdirs();
    
    String pathData = dir.getPath()+File.separator+sievename+"-data.ser";
    String pathModel = dir.getPath()+File.separator+sievename+"-model.ser";
    if((new File(pathModel)).exists()) {
      System.err.println(sievename+" model exist!");
    } else {
      // data extraction
      RVFDataset<Boolean, String> traindata = null;
      if((new File(pathData)).exists()) {
        traindata = IOUtils.readObjectFromFile(pathData);
      } else {
        props.put(CorefProperties.PATH_INPUT_PROP, CorefProperties.getPathTrainData(props)); 
        traindata = extractTrainData(props, sievename);
        System.err.println("traindata.size(): "+traindata.size());
        System.err.println("traindata.numFeatures(): "+traindata.numFeatures());
        traindata.applyFeatureCountThreshold(5);    // to reduce filesize (will be done again in train())
        
        if(CorefProperties.storeTrainData(props)) IOUtils.writeObjectToFile(traindata, pathData);
      }
      System.err.println("after applyFeatureCountThreshold traindata.size(): "+traindata.size());
      System.err.println("after applyFeatureCountThreshold traindata.numFeatures(): "+traindata.numFeatures());
      
      if(CorefProperties.doPMIFeatureSelection(props, sievename)) {
        Counter<String> pmi = MutualInformationFeatureSelection.featureSelectionByMutualInfo(traindata);
        double pmiThres = CorefProperties.getPMIThres(props, sievename);
        Counters.retainAbove(pmi, pmiThres);
        traindata = MutualInformationFeatureSelection.retainNumericOrSelected(traindata, pmi.keySet());
      }
      System.err.println("after feature selection traindata.size(): "+traindata.size());
      System.err.println("after feature selection traindata.numFeatures(): "+traindata.numFeatures());
      
      Sieve sieve = null;
      sieve = boostTrain(traindata, pathModel, props, sievename);
      IOUtils.writeObjectToFile(sieve, pathModel);
    }
  }
}
