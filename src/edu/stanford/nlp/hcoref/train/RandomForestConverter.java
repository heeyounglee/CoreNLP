package edu.stanford.nlp.hcoref.train;

import hr.irb.fastRandomForest.FastRandomForest;
import hr.irb.fastRandomForest.FastRandomTree;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import edu.stanford.nlp.hcoref.rf.DecisionTree;
import edu.stanford.nlp.hcoref.rf.DecisionTreeNode;
import edu.stanford.nlp.hcoref.rf.RandomForest;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;

/** convert FastRandomForest format to our own RandomForest */
public class RandomForestConverter {
  public static RandomForest convert(FastRandomForest fastrf) {
    
    Index<String> featureIndex = makeFeatureIndex(fastrf);
    
    RandomForest rf = new RandomForest(featureIndex, fastrf.getNumTrees());
    
    // copy each tree
    int i=0;
    for (Classifier c : fastrf.m_bagger.getClassifiers()) {
      FastRandomTree rtree = (FastRandomTree) c;
      DecisionTree copiedTree = convert(rtree, featureIndex);
      rf.trees[i++] = copiedTree;
    }
    return rf;
  }
  private static Index<String> makeFeatureIndex(FastRandomForest frf) {
    Instances instances = frf.m_Info;
    Index<String> featureIndex = new HashIndex<String>();
    
    for (int i=0 ; i < instances.numAttributes()-1 ; i++) {
      if(instances.classIndex() == i) continue;
      Attribute attr = instances.attribute(i);
      featureIndex.add(attr.name());
    }
    return featureIndex;
  }
  private static DecisionTree convert(FastRandomTree rtree, Index<String> featureIndex) {
    DecisionTree tree = new DecisionTree(featureIndex);
    tree.root = convert(rtree);
    return tree;
  }
  
  /** Recursive conversion */
  private static DecisionTreeNode convert(FastRandomTree rtree) {
    if(rtree.m_Attribute < 0) {    // node is a leaf
      Attribute cAttr = rtree.m_MotherForest.m_Info.classAttribute();
      int trueIdx = cAttr.indexOfValue("true");
      int falseIdx = cAttr.indexOfValue("false");
      double trueProb = rtree.m_ClassProbs[trueIdx];
      double falseProb = rtree.m_ClassProbs[falseIdx];
      
      int label = (trueProb > falseProb)? 1 : 0;
      return new DecisionTreeNode(label, (float) (trueProb / (trueProb + falseProb)));
    }
    DecisionTreeNode left = convert(rtree.m_Successors[0]);
    DecisionTreeNode right = convert(rtree.m_Successors[1]);
    
    DecisionTreeNode node = new DecisionTreeNode(rtree.m_Attribute, (float) rtree.m_SplitPoint, new DecisionTreeNode[]{left, right});
    
    return node;
  }
  
  public static void main(String[] args) throws Exception {
//    MentionDetectionClassifier md = IOUtils.readObjectFromFile("/220/cleanup/ser-small/md-model.ser");
//    RVFDataset<Boolean, String> data = IOUtils.readObjectFromFile("/220/cleanup/ser-small/md-data.ser");
//    
//    FastRandomForest frf = md.rf;
//    Instances instances = frf.m_Info;
//    RandomForest rf = convert(frf);
//    
//    Attribute cAttr = frf.m_Info.classAttribute();
//    int trueIdx = cAttr.indexOfValue("true");
//    int falseIdx = cAttr.indexOfValue("false");
//    
//    double max = 0;
//    
//    for (RVFDatum<Boolean, String> datum : data) {
//      Instance ins = DataConverter.makeInstance(datum, instances, true);
//      double pTrue = rf.probabilityOfTrue(datum);
//      double[] probs = frf.distributionForInstance(ins);
//      double probTrue = probs[trueIdx];
//      
//      double diff = Math.abs(pTrue - probTrue);
//      if(diff > max) max = diff;
//      System.err.println(diff);
//      System.err.println("------------------------");
//    }
//    System.err.println(max);
//    
//    System.err.println();
//    System.err.println(md.rf);
  }
}
