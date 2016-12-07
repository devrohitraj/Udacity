Classification
==============

- Calculation time : Order
- Application


Decision Tree
-------------

- tree.DecisionTreeClassifier

Decision tree advantages
- Simple to understand and interpret
- Requires little data preparation
- able to handle both numerical and categorical data

Limitations
- learning an optimal decision tree is known to be NP-complete
- tend to overfit
  
Neural Networks
---------------

- neural_network.BernoulliRBM
  Bernoulli Restricted Boltzmann Machine (RBM)

neural_network * logistic

Instance based Learning
-----------------------

- neighbors.KNeighborsClassifier

SVM
---

- svm.SVC
  Support Vector Machine algorithm

Naive Bayes
-----------

Naive bayes is typically used for text classification like spam
filter. It is simple technique for constructing classifiers and
assumes that the value of a particular feature is indepenendent
regardless of any possible correlations between features. Advantage of
Naive bayes is that it only requires a small amount of trainig data to
estimate the parameters necessary for classification.

- According to Naive Bayes's concept, you should not mix dicrete and
  continuous variables in dataset.
  - Gaussian naive bayes: deal with continuous data
  - Multinomial naive bayes: feature vectors represent the frequencies
  - Bernoulli naive bayes: features are independent booleans(binary values)

- lib: naive_bayes.MultinomialNB, naive_bayes.BernoulliNB
  
Ensamble
--------

Random forest is one of the ensamble method which randomly combines
decision trees to improve accuracy to test data. It avoids overfitting
by combining multiple decision trees.
