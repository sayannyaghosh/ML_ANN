import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import sklearn
from sklearn import preprocessing
# from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
#from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from mlxtend.plotting import plot_linear_regression

from mlxtend.evaluate import confusion_matrix 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc

conv_dat=[]
xx=[]
inpt_test=[]
training_data=[]
testing_data=[]

training_data=pd.read_csv('/home/sayannya/Desktop/SGANNDD/training.csv')
testing_data=pd.read_csv('/home/sayannya/Desktop/SGANNDD/testing.csv')

X=preprocessing.normalize(training_data.iloc[:, [0,1,2,3]])  #Features of Drugs used for training
y=preprocessing.normalize(training_data.iloc[:, [4]]) #Actual Class  used for training

# scale units
#X= X/np.amax(data, axis=0) # maximum of X array
#y= y/2 # max test score is 2


#--------------------------
#Training Phase start here|
#--------------------------
class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 4
    self.outputSize = 1
    self.hiddenSize = 3
    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x3) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
    
  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o 


  def sigmoid(self, s):
    # activation function 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

NN = Neural_Network()
max_ite=100
for i in xrange(max_ite): # trains the NN 1,000 times

  merror=np.mean(np.square(y - NN.forward(X)))
  
  conv_dat.append(merror)
  xx.append(i)
  print "Training error: \n" + str(merror)
  #print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
  print "\n"
  NN.train(X, y)

#-------------------------
#Training phase end here |
#-------------------------


#--------------------------------------
#Prediction(Testing Phase) start here |
#--------------------------------------

inpt_test=preprocessing.normalize(testing_data.iloc[:, [0,1,2,3]]) #Data used for Testing

Act_op=preprocessing.normalize(testing_data.iloc[:,[4]]) #Actual Output for Testing

predicted_op=NN.forward(inpt_test) #Predicted Class
#print('Actual output :',Act_op,'Predicted output by ANN ', predicted_op)
print('[1]-> Active , [0]-> Inactive ')
err=np.mean(np.square(Act_op - predicted_op))

print('Testing error  :',err)

#-------------------------------------
#Prediction(Testing Phase) end here  |
#-------------------------------------


#--------------------------------------
#Precision Recall F-Measure done here |
#--------------------------------------
average_precision = average_precision_score(Act_op, predicted_op)
print('Precision Score is:',average_precision*100)
 
accScore=accuracy_score(np.array(Act_op), np.array(predicted_op.round()))
print('Accuracy Score :',accScore*100)
print('Recall Score :',recall_score(Act_op, predicted_op.round())*100 )
print('Report : ')
print(classification_report(Act_op, predicted_op.round()))
#--------------------------------------
#Precision Recall F-Measure done here |
#--------------------------------------


#----------------------------
#Confusion Matrix Plot Here |
#----------------------------
print('Confusion Matrix :')
cm = confusion_matrix(y_target=Act_op, 
                      y_predicted=predicted_op.round())
print(cm)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()
#--------------------------------
#Confusion Matrix Plot End Here |
#--------------------------------

#----------------------
#ROC curve start here |
#----------------------

print(Act_op)
print('------------------------------------')
print(predicted_op)

fpr, tpr, thresholds = roc_curve(Act_op, predicted_op)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=0.5, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


#--------------------
#ROC curve end here |
#--------------------


#-----------------------------
#Convergence Plot start here |
#-----------------------------
plt.semilogy(xx, conv_dat,color='red', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=5) 
plt.grid(True)
  
# naming the x axis 
plt.xlabel('Iteration') 
# naming the y axis 
plt.ylabel('MSE') 
  
# giving a title to my graph 
plt.title('Active and Inactive molecule classification in Drug prediction convergence plot') 

# function to show the plot 
plt.show()

#----------------------
#Convergence Plot end |
#----------------------