#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
from requests_oauthlib.compliance_fixes.slack import slack_compliance_fix
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

import numpy as np
### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

data_dict.pop("TOTAL", 0)

features = ["salary", "bonus"]

data = featureFormat(data_dict, features)


### your code below


'''from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
print reg.score(ages_test, net_worths_test)

print 'Coefficient', reg.coef_
print 'Intercept', reg.intercept_

predicted = reg.predict(ages_test)'''



for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
maximum = 0
i = 0
j = 0
for point in data:
    i = i+ 1
    #print i, point[0]
    if point[0] > 1000000:
        
        print point[0]
'''print data_dict
k = -1        
for i in data_dict:
    k = k + 1
    print i
    if data_dict[i]['salary'] == maximum:
        print k
        del data_dict[j-1]
        print 'yes'
        
        break'''
        
        
print 'max is',maximum
print 'index is', j

#print len(features[0])
#data = featureFormat(data_dict, features)


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

