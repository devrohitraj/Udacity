
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project 2: Building a Student Intervention System

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ### Question 1 - Classification vs. Regression
# *Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

# **Answer: **

# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.

# In[79]:

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
#print student_data
print "Student data read successfully!"


# ### Implementation: Data Exploration
# Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
# - The total number of students, `n_students`.
# - The total number of features for each student, `n_features`.
# - The number of those students who passed, `n_passed`.
# - The number of those students who failed, `n_failed`.
# - The graduation rate of the class, `grad_rate`, in percent (%).
# 

# In[80]:

# TODO: Calculate number of students
n_students = student_data.shape[0]

# TODO: Calculate number of features
n_features = student_data.shape[1]

# TODO: Calculate passing students
count = 0
for student in student_data['passed']:
    if student == 'yes':
        count = count + 1
n_passed = count

# TODO: Calculate failing students
n_failed = n_students - n_passed

# TODO: Calculate graduation rate
grad_rate = (n_passed/float(n_students)) * 100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# ## Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
# 
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
# 
# Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.

# In[81]:

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


# ### Preprocess Feature Columns
# 
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.

# In[82]:

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
#print X_all.shape[0]
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))


# ### Implementation: Training and Testing Data Split
# So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
# - Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
#   - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
#   - Set a `random_state` for the function(s) you use, if provided.
#   - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.

# In[83]:

# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split
# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size= num_test, train_size=num_train, random_state=42)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# ## Training and Evaluating Models
# In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.

# ### Question 2 - Model Application
# *List three supervised learning models that are appropriate for this problem. What are the general applications of each model? What are their strengths and weaknesses? Given what you know about the data, why did you choose these models to be applied?*

# **Answer: Decision Tree, KNN, SVM
# Desion Tree: Fast training and Accuracy
# **

# ### Setup
# Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
# - `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
# - `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
# - `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
#  - This function will report the F<sub>1</sub> score for both the training and testing data separately.

# In[85]:

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Implementation: Model Performance Metrics
# With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
#  - Use a `random_state` for each model you use, if provided.
#  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Create the different training set sizes to be used to train each model.
#  - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
# - Fit each model with each training set size and make predictions on the test set (9 in total).  
# **Note:** Three tables are provided after the following code cell which can be used to store your results.

# In[88]:

# TODO: Import the three supervised learning models from sklearn
from sklearn import svm
from sklearn import tree
from sklearn.neighbors.classification import KNeighborsClassifier

# TODO: Initialize the three models
clf_A = svm.SVC()
clf_B = tree.DecisionTreeClassifier()
clf_C = KNeighborsClassifier(n_neighbors=3)

# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
print "------------------------------------------"
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
print "------------------------------------------"
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)


# ### Tabular Results
# Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

# ** Classifer 1 - SVM**  
# 
# | Training Set Size | Prediction Time (train) | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |    0.0010 seconds       |     0.0010 seconds     |    0.8777        |    0.7746       |
# | 200               |    0.0030               |     0.0010             |    0.8679        |    0.7815       |
# | 300               |    0.0070               |     0.0030             |    0.8761        |    0.7838       |
# 
# ** Classifer 2 - Decision Tree**  
# 
# | Training Set Size | Prediction Time (train) | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |     0.0010              |     0.0000             |    1.0000        |    0.6140       |
# | 200               |     0.0010              |     0.0000             |    1.0000        |    0.7077       |
# | 300               |     0.0010              |     0.0000             |    1.0000        |    0.5950       |
# 
# ** Classifer 3 - KNN**  
# 
# | Training Set Size | Prediction Time (train) | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |     0.0140              |     0.0020             |     0.8321       |     0.7482      |
# | 200               |     0.0050              |     0.0040             |     0.8966       |     0.7413      |
# | 300               |     0.0100              |     0.0040             |     0.8807       |     0.7313      |

# ## Choosing the Best Model
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score. 

# ### Question 3 - Chosing the Best Model
# *Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

# **Answer: Decision Tree is the best model i tested here. It takes less time to train and predict data and it has higher accuracy than others**

# ### Question 4 - Model in Layman's Terms
# *In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. For example if you've chosen to use a decision tree or a support vector machine, how does the model go about making a prediction?*

# **Answer: Decision tree is the best model i used here. It takes less time and has very good accuracy on predicting things.
#  This model makes trees of conditions from given data and then based on condition given it go on and predict outcomes. So, by this way we don't have to waste out time analysing unnesserary data that's why it is fast and also more accurate**

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
# - Initialize the classifier you've chosen and store it in `clf`.
# - Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
#  - Set the `pos_label` parameter to the correct value!
# - Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.

# In[89]:

# TODO: Import 'gridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from  sklearn.metrics import make_scorer
from sklearn import metrics
# TODO: Create the parameters list you wish to tune
parameters_a = {'gamma' : [0, 10, 100]}

parameters_b = {'max_depth' : [1,2,3,4,5,6]}

parameters_c = {  'n_neighbors' : (3, 5, 10, 15, 20, 25, 30, 35, 40, 50),
                'weights'     : ('uniform', 'distance'),
                'p'           : (1, 2, 3)}


parameters = parameters_b

# TODO: Initialize the classifier
clf_A = svm.SVC()
clf_B = tree.DecisionTreeClassifier()
clf_C = KNeighborsClassifier(n_neighbors=3)

clf = clf_B

# TODO: Make an f1 scoring function using 'make_scorer'

clf_scorer = metrics.make_scorer(metrics.f1_score, pos_label='yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=clf_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print clf
#print grid_obj.scorer_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Question 5 - Final F<sub>1</sub> Score
# *What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

# **Answer: Final score is much accurate than un-tuned ones.**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
