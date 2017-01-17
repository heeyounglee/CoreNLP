package edu.stanford.nlp.hcoref.train;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.util.Generics;

// GeneralDataset -> Instances (weka format)
public class DataConverter {
  
  // attributes info is not given, so make it from data
  public static <L> Instances convertGeneralDatasetToInstances(GeneralDataset<L, String> dataset, boolean useSparseInstance) {
    
    // get attributes info
    int numAttributes = dataset.numFeatures()+1;    // label is an attribute
    FastVector attrInfo = new FastVector(numAttributes);
    Map<String, Attribute> attrIndex = new HashMap<String, Attribute>();
    
    Attribute attr;
    
    for(String feature : dataset.featureIndex()){
      
      // boolean feature
//      FastVector v = new FastVector();
//      v.addElement("False");
//      v.addElement("True");
//      attr = new Attribute(feature, v);

      // numeric feature
      attr = new Attribute(feature);    // use numeric feature

      attrIndex.put(feature, attr);
      attrInfo.addElement(attr);
    }
    
    // add class attribute at last
    FastVector fv = new FastVector();
    for(L m : dataset.labelIndex()) {
      fv.addElement(m.toString());
    }
    if(fv.size()==1) fv.addElement("true");   // TODO temp for debug
    attr = new Attribute("Label", fv);
    attrIndex.put("Label", attr);
    attrInfo.addElement(attr);
    
    Instances instancesForAttrInfo = new Instances("dataset", attrInfo, dataset.size());
    instancesForAttrInfo.setClassIndex(numAttributes-1);
    
    return convertGeneralDatasetToInstances(dataset, instancesForAttrInfo, useSparseInstance);
  }
  
  // given attributes info (instancesForAttrInfo), convert data -> Instances
  public static <L> Instances convertGeneralDatasetToInstances(
      GeneralDataset<L, String> dataset, 
      Instances instancesForAttrInfo, 
      boolean useSparseInstance) {
    
    Instances instances = new Instances(instancesForAttrInfo, dataset.size());
    
    for(RVFDatum<L, String> datum : dataset) {
      Instance i = makeInstance(datum, instances, useSparseInstance);
      i.setDataset(instances);
      instances.add(i);
    }
    return instances;
  }

  public static <L> Instance makeInstance(RVFDatum<L, String> datum, 
      Instances instancesForAttrInfo,
      boolean useSparseInstance) {
    
    int numAttributes = instancesForAttrInfo.numAttributes();
    Map<String, Attribute> attrIndex = new HashMap<String, Attribute>();
    for(int idx=0 ; idx<numAttributes ; idx++){
      Attribute attr = instancesForAttrInfo.attribute(idx);
      attrIndex.put(attr.name(), attr);
    }
      
    Attribute targetAttr = instancesForAttrInfo.classAttribute();
    
    double[] attValues = new double[numAttributes];
    Instance instance = (useSparseInstance)? new SparseInstance(1.0, attValues) : new DenseInstance(1.0, attValues);
    
    // label
    instance.setValue(instancesForAttrInfo.classAttribute(), datum.label().toString() );
    
    // following loop is over *defined* features
    // datum may contain additional features which are ignored here
    for (String feature : datum.asFeatures()) {
      if(!attrIndex.containsKey(feature)) {
        continue;
      }
      Object val = datum.getFeatureCount(feature);
      if(attrIndex.get(feature)==targetAttr) val = datum.label();
      
      if (val instanceof Boolean) val = val.toString();
      Attribute attr = attrIndex.get(feature);
      if (attr == null)
        throw new RuntimeException("Unknown feature: " + feature);
      try {
        if (val instanceof String) {
          // System.err.println("feature = val: " + feature + " = " + val);
          instance.setValue(attr, (String) val);
        } else if (val instanceof Double) {
          instance.setValue(attr, (Double) val);
        } else if (val instanceof Integer) {
          instance.setValue(attr, new Double((Integer) val));
        } else if (val != null) {
          instance.setValue(attr, val.toString());
        }
      } catch (IllegalArgumentException e) {
        throw new IllegalArgumentException("For " + feature + " = " + val + " with type " + val.getClass() + ":\n" + e);
      }
    }
    if(useSparseInstance) return new SparseInstance(instance);
    return instance;
  }
  
  public static void filterData(Instances instances, String fileteredStr) {
    List<Integer> toDelete = new ArrayList<Integer>();
    for(Enumeration<Attribute> enu = instances.enumerateAttributes(); enu.hasMoreElements();) {
      Attribute attribute = enu.nextElement();
      
      // filter here
      if(attribute.name().contains(fileteredStr)) toDelete.add(attribute.index());
    }
    for(int idx=toDelete.size()-1 ; idx >= 0 ; idx--){
      instances.deleteAttributeAt(toDelete.get(idx));
    }
  }
  
  // 
  public static void printDataMatrix(List<String> features, GeneralDataset<Boolean, String> dataset, String fileFeatures, String fileLabels) {
    try {
      PrintWriter pw = IOUtils.getPrintWriter(fileFeatures);
      PrintWriter pwLabel = IOUtils.getPrintWriter(fileLabels);
      for(RVFDatum<Boolean, String> datum : dataset){
        StringBuilder sb = new StringBuilder();
        for(String f : features){
          double value = datum.getFeatureCount(f);
          sb.append(value).append("\t");
        }
        sb.append("\n");
        pwLabel.println(datum.label());
        pw.print(sb.toString());
      }
      pw.close();
      pwLabel.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  public static void printDataRow(List<String> features, RVFDatum<Boolean, String> datum, String fileFeatures, String fileLabels) {
//    try {
//      File pw = new File(fileFeatures);
//      File pwLabel = new File(fileLabels);
      StringBuilder sb = new StringBuilder();
      for(String f : features){
        double value = datum.getFeatureCount(f);
        sb.append(value).append("\t");
      }
      sb.append("\n");
      System.err.println("MATLAB-DATA:"+sb.toString());
      System.err.println("MATLAB-LABEL:"+datum.label());
//      IOUtils.writeObjectToFile(datum.label().toString(), pwLabel, true);
//      IOUtils.writeObjectToFile(sb.toString(), pw, true);
//    } catch (IOException e) {
//      throw new RuntimeException(e);
//    }
  }
  public static List<String> featureList(GeneralDataset<Boolean, String> dataset, String file){
    try {
      PrintWriter pw = IOUtils.getPrintWriter(file);
      List<String> features = new ArrayList<String>();
      for(String feature : dataset.featureIndex){
        features.add(feature);
        pw.println(feature);
      }
      pw.close();
      return features;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  public static List<String> loadFeatureList(String file){
    List<String> features = Generics.newArrayList();
    for(String line : IOUtils.readLines(file)) {
      features.add(line);
    }
    return features;
  }

  public static void main(String[] args) throws Exception {

//    Instances i = (new DataSource("/home/heeyoung/hcoref/temp.arff")).getDataSet();
//    i.setClassIndex(i.numAttributes() - 1);
//    IOUtils.writeStringToFileNoExceptions(i.toString(), "/home/heeyoung/hcoref/tempnew.arff", "UTF-8");
    
    // GeneralDataset to double matrix for matlab
    RVFDataset<Boolean, String> traindata = IOUtils.readObjectFromFile("/home/heeyoung/log-hcoref/data/default/rf_pronoun_generaldataset.ser");
    List<String> features = DataConverter.featureList(traindata, 
        "/home/heeyoung/log-hcoref/data/default/rf_pronoun_matrix_features.dat");
    DataConverter.printDataMatrix(features, traindata, 
        "/home/heeyoung/log-hcoref/data/default/rf_pronoun_matrix-train.dat", 
        "/home/heeyoung/log-hcoref/data/default/rf_pronoun_matrix_labels-train.dat");

  }
}
