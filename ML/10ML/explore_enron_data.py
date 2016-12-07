#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
len = 0
enron_data1 = open("../final_project/poi_names.txt", "r")
print enron_data


#print enron_data1
#for data in enron_data:
 #   print data + str(len)
  #  len = len + 1
    
#print len
#for data in enron_data:
 #   if enron_data[data]['poi'] is True:
  #      len = len + 1
    
#print enron_data['PRENTICE JAMES']['total_stock_value']
name = 'LAY KENNETH L'.upper()
feature = 'total_payments'
count_payment = 0
count_sal = 0
tot_count = 0

for i in enron_data:
    
    if enron_data[i]['poi'] is True:
        tot_count = tot_count + 1
        if enron_data[i]['total_stock_value'] == 'NaN':
            count_payment = count_payment + 1
        
    '''if enron_data[i]['salary'] != 'NaN':
        count_sal = count_sal + 1 '''   
        
print 'Total_ Payment NaN is', count_payment
print 'Total no. is', tot_count
print 100/28
#print 'Quantified Email-ID is', count_email


#print enron_data[name][feature]

