package edu.stanford.nlp.hcoref.train;

import java.util.Properties;

import hr.irb.fastRandomForest.FastRandomForest;
import edu.stanford.nlp.hcoref.CorefProperties;
import edu.stanford.nlp.hcoref.md.MentionDetectionClassifier;
import edu.stanford.nlp.hcoref.sieve.Sieve.ClassifierType;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Redwood;

public class CorefTrainer {

  public static void trainCorefSystem(Properties props) throws Exception {
    
    // train md
    Redwood.log("coref-train", "train mention span classifier...");
    MentionDetectionClassifierFactory.trainMentionSpanClassifier(props);
    
    // train sieves
    String[] sievenames = CorefProperties.getSieves(props).split(",");
    for (String sievename : sievenames) {
      sievename = sievename.trim();
      ClassifierType type = CorefProperties.getClassifierType(props, sievename);
      switch (type) {
        case RF:
          Redwood.log("coref-train", "train "+sievename+" ...");
          props.put(CorefProperties.CURRENT_SIEVE_FOR_TRAIN_PROP, sievename);
          RFSieveFactory.trainSieve(props, sievename);
          break;
        case LINEAR:
	      Redwood.log("coref-train", "train "+sievename+" ...");
	      props.put(CorefProperties.CURRENT_SIEVE_FOR_TRAIN_PROP, sievename);
	      LinearSieveFactory.trainSieve(props, sievename);
	      break;        	
        case ORACLE:
        case RULE:
        default:
      }
    }
    
  }

  public static void main(String[] args) throws Exception {
    Redwood.hideChannelsEverywhere(
        "debug-cluster", "debug-mention", "debug-preprocessor", "debug-docreader", "debug-mergethres",
        "debug-featureselection", "debug-md"
        );
    
    Properties props = StringUtils.argsToProperties(args);
    trainCorefSystem(props);
  }
  
  public static void printRFFeatureImportance(FastRandomForest rf) {
    System.err.println("Feature importances - increase in out-of-bag error (as % misclassified instances) after feature permuted:\n");
    double[] importances = rf.m_bagger.getFeatureImportances();
    for ( int i = 0; i < importances.length; i++ ) { 
      System.err.println( String.format( "%d\t%s\t%6.4f%%\n", i+1, rf.m_Info.attribute(i).name(),
              i==rf.m_Info.classIndex() ? Double.NaN : importances[i]*100.0 ) ); //bagger.getFeatureNames()[i] );
    }
  }
}
