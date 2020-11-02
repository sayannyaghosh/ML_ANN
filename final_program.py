#------------------------------
# importing libraries
#------------------------------
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate

#------------------
# dataset Reading
#------------------
df= pd.read_csv('data.csv') #dataframe
X = df.iloc[:,range(0,4)] #Feature set <--: all row, 
y = df.iloc[:,4] #Class labels <--: active class and inactive class

#------------------------------------------
# MLP Classifier Design
#------------------------------------------
clf = MLPClassifier(activation='logistic',hidden_layer_sizes=(10,),
                    max_iter=500,solver='sgd',batch_size=1)

#-----------------------------------
# Performance metric initialization 
#-----------------------------------
scr = ['accuracy', 'f1', 'precision', 'recall']

#--------------------------------------
# Train MLP model and test 5-Fold  
#--------------------------------------
k = 5 #no. of folds 
scores = cross_validate(clf, X, y, cv=k, scoring=scr)

#-----------------------------
# Model's performance storing
#-----------------------------

test_accuracy = scores['test_accuracy'] #Test accuracy
test_f1 = scores['test_f1'] #Test f1 score
test_pre = scores['test_precision'] #Test Precision
test_rec = scores['test_recall'] #Test recall

#-----------------------------------
# Printing the results of MLP model
#-----------------------------------
print("Test accuracy:",(sum(test_accuracy)/k)*100)
print("Test F1 score:", (sum(test_f1)/k))
print("Test Precision:", (sum(test_pre)/k)*100)
print("Test Recall:", (sum(test_rec)/k)*100)
