Hybrid coreference resolution system
====================================

The code is based on Stanford CoreNLP and FastRandomForest (+ weka),
forked from:

https://github.com/stanfordnlp/CoreNLP

https://code.google.com/archive/p/fast-random-forest/

How to run
----------

* Train
```
java -Xmx30g edu.stanford.nlp.hcoref.train.CorefTrainer
    -props /PATH/src/edu/stanford/nlp/hcoref/properties/coref-conll.properties \
    -hcoref.path.serialized /PATH/MODEL/SERIALIZED/
```

* Eval
```
java -Xmx30g edu.stanford.nlp.hcoref.CorefSystem
    -props /PATH/src/edu/stanford/nlp/hcoref/properties/coref-conll.properties \
    -hcoref.path.serialized /PATH/MODEL/SERIALIZED/
```
