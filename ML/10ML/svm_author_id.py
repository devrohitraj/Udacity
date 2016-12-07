#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

## Reduced the Training dataset to 1%
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


## Training and predicting dataset
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
prediction = clf.predict(features_test)
count = 0
for i in range(len(prediction)):
    if prediction[i] == 1:
        count = count + 1
print count   
'''print prediction[10]
print prediction[26]
print prediction[50]'''
print "prediction time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, prediction)
#########################################################


