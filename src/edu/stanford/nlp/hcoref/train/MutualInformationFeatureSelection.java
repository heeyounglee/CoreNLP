package edu.stanford.nlp.hcoref.train;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.TwoDimensionalCounter;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

public class MutualInformationFeatureSelection<L,F> {

  public static <L,F> void mutualInformation(GeneralDataset<L,F> data, TwoDimensionalCounter<L,F> mi, TwoDimensionalCounter<L,F> nPMI) {
    
    // 2dCounter<Feature, Class>
    TwoDimensionalCounter<F, L> featureClassCounter = new TwoDimensionalCounter<F,L>();
    Counter<L> label = new ClassicCounter<L>();
    
    for(RVFDatum<L, F> datum : data) {
      L l = datum.label();
      label.incrementCount(l);
      
      Counter<F> fs = datum.asFeaturesCounter();
      for(F f : fs.keySet()) {
        if(fs.getCount(f)>0) featureClassCounter.incrementCount(f, l);
      }
    }
    
    for(L l : data.labelIndex) {
      for(F f : data.featureIndex) {
        double nfl = featureClassCounter.getCount(f, l);
        double nf = featureClassCounter.getCounter(f).totalCount();
        double nl = label.getCount(l);
        
        double n11 = nfl;
        double n10 = nf - nfl;
        double n01 = nl - nfl;
        double n00 = data.size() - n10 - n01 - n11;
        
        double n = n11+n10+n01+n00;
        double n1dot = n11+n10;
        double ndot1 = n01+n11;
        double n0dot = n01+n00;
        double ndot0 = n00+n10;
        
        double muInfo = 0;
        double npmi = Math.log((n*n11)/(n1dot*ndot1))/Math.log(2)/(Math.log(n/n11)/Math.log(2));
        
        if(n11!=0) muInfo += n11/n*Math.log((n*n11)/(n1dot*ndot1))/Math.log(2);
        if(n10!=0) muInfo += n10/n*Math.log((n*n10)/(n1dot*ndot0))/Math.log(2);
        if(n01!=0) muInfo += n01/n*Math.log((n*n01)/(n0dot*ndot1))/Math.log(2);
        if(n00!=0) muInfo += n00/n*Math.log((n*n00)/(n0dot*ndot0))/Math.log(2); 
        
        mi.setCount(l, f, muInfo);
        nPMI.setCount(l, f, npmi);
        
      }
    }
  }
  
  public static Counter<String> featureSelectionByMutualInfo(RVFDataset<Boolean, String> data) {
    TwoDimensionalCounter<Boolean, String> selected = new TwoDimensionalCounter<Boolean, String>();
    TwoDimensionalCounter<Boolean, String> nPMI = new TwoDimensionalCounter<Boolean, String>();
    
    mutualInformation(data, selected, nPMI);
    
    Counter<String> s = selected.getCounter(true);
    List<Pair<String, Double>> up = Counters.toSortedListWithCounts(s);

    // counts for feature occurrences
    TwoDimensionalCounter<String, Boolean> featureClassCounter = new TwoDimensionalCounter<String, Boolean>();
    Counter<Boolean> label = new ClassicCounter<Boolean>();
    for(RVFDatum<Boolean, String> datum : data) {
      boolean l = datum.label();
      label.incrementCount(l);
      
      Counter<String> fs = datum.asFeaturesCounter();
      for(String f : fs.keySet()) {
        if(fs.getCount(f)>0) featureClassCounter.incrementCount(f, l);
      }
    }
    
    DecimalFormat df = new DecimalFormat("##.####");
    Redwood.log("debug-featureselection", "FEATURE SELECTION --------------------------------");
    for(Pair<String, Double> pair : up) {
      double featureCountTrue = featureClassCounter.getCount(pair.first(), true);
      double featureCountFalse = featureClassCounter.getCount(pair.first(), false);
      double npmiTrue = nPMI.getCount(true, pair.first());
      double npmiFalse = nPMI.getCount(false, pair.first());
      
      String feature = pair.first();
      double mutualInfo = pair.second();
      
      StringBuilder sb = new StringBuilder();
      sb.append(feature).append("\t");
      sb.append(df.format(mutualInfo)).append("\t");
      sb.append(df.format(featureCountTrue)).append("\t");
      sb.append(df.format(featureCountFalse)).append("\t");
      sb.append(df.format(npmiTrue)).append("\t");
      sb.append(df.format(npmiFalse));
      
      Redwood.log("debug-featureselection", sb.toString());
    }
    Redwood.log("debug-featureselection", "FEATURE SELECTION --------------------------------");
    return s;
  }
  
  // remove features from dataset
  public static RVFDataset<Boolean, String> retainNumericOrSelected(RVFDataset<Boolean, String> data, Set<String> selected) {
    RVFDataset<Boolean, String> modifiedData = new RVFDataset<Boolean, String>();
    for(RVFDatum<Boolean, String> datum : data) {
      Counter<String> features = datum.asFeaturesCounter();
      Set<String> remove = Generics.newHashSet();
      for(String f : features.keySet()) {
        if(f.startsWith("B-") && !selected.contains(f)) {
          remove.add(f);
        }
      }
      Counters.removeKeys(features, remove);
      modifiedData.add(new RVFDatum<Boolean, String>(features, datum.label()));
    }
    return modifiedData;
  }
}
