#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.neighbors.classification import KNeighborsClassifier
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

### For KNN ###
#from sklearn.neighbors import NearestNeighbors
#clf = KNeighborsClassifier()


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, prediction)

#########################################################


