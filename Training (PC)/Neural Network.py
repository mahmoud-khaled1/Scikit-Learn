#Note :SkLearn have Simple Neural network algorithms so we will learn it in details in " ML " in TensorFlow Library

#1-MLP Regressor  for regression

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#Set DataSet
Boston=load_boston()
X=Boston.data
Y=Boston.target

#Splitting data set
X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True,random_state=44)

#Appltying MLPRegressor algorithm

MLPRegressorModel=MLPRegressor(hidden_layer_sizes=(100,3),activation="relu",solver="lbfgs",alpha=0.0001,
                               learning_rate='constant',early_stopping=False,random_state=44,max_iter=10000)
MLPRegressorModel.fit(X_train,y_train)

#Show Details
print("train Score :",MLPRegressorModel.score(X_train,y_train))
print("Test Score :",MLPRegressorModel.score(X_test,y_test))
print("No of Iteration :",MLPRegressorModel.n_iter_)
print("No of Layer :",MLPRegressorModel.n_layers_)

Y_pre=MLPRegressorModel.predict(X_test)

print(list(Y_pre[:10]))
print(list(y_test[:10]))


MAE_Value=mean_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAE_Value)


MAESqu_Value=mean_squared_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAESqu_Value)


median_value =median_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(median_value)

#----------------------------------------------------------
#2-MLP Classifier  for Classification

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

#Set DataSet
Boston=load_iris()
X=Boston.data
Y=Boston.target

#Splitting data set
X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True,random_state=44)

#Appltying MLPRegressor algorithm

MLPClassifierModel=MLPClassifier(hidden_layer_sizes=(100,3),activation="relu",solver="lbfgs",alpha=0.0001,
                               learning_rate='constant',early_stopping=False,random_state=44,max_iter=10000)


MLPClassifierModel.fit(X_train,y_train)

#Show Details
print("train Score :",MLPClassifierModel.score(X_train,y_train))
print("Test Score :",MLPClassifierModel.score(X_test,y_test))
print("No of Iteration :",MLPClassifierModel.n_iter_)
print("No of Layer :",MLPClassifierModel.n_layers_)

Y_pre=MLPClassifierModel.predict(X_test)

print(list(Y_pre[:10]))
print(list(y_test[:10]))


#Calculted Coufusion matrix
#[TP     FP]
#[FN     TN]
##We should make TP and TN biggest and smallest FP and FN
CM=confusion_matrix(y_test,Y_pre)
print("Confusion Matrix :",CM)