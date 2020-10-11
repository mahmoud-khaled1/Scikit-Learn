#Spliting Data To Training data and Testing Data

#train_test_split : use in 90% in splitting data in projects .
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

Cancer_data=load_breast_cancer()
X=Cancer_data.data
Y=Cancer_data.target
print("Cancer Data : ",X.shape,Y.shape)
print("*"*50)
#Test Size range (0-1) if test_size==0.2 and data is 1000 row then train data=800 and test data=200
#random_state is any number 44,87,5454,.....  you want .
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=44,shuffle=True)

print("X_train:",X_train.shape)
print("X_test:",X_test.shape)
print("Y_train:",y_train.shape)
print("y_test:",y_test.shape)
#-----------------------------------------------------------------

#KFold
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

Cancer_data=load_breast_cancer()
X=Cancer_data.data
Y=Cancer_data.target
print("Cancer Data : ",X.shape,Y.shape)
print("*"*100)

#split data to 4 parts
KF=KFold(n_splits=4,shuffle=True,random_state=44)

#print Data after Splitting them
for train_index,test_index in KF.split(X):
    print("Train: " ,train_index, "Test: ",test_index)
    X_train,X_test=X[train_index],X[test_index]
    y_train, y_test = Y[train_index],Y[test_index]
    print("X_Train \n",X_train)
    print("X_Test \n",X_test)
    print("Y_Train \n",y_train)
    print("Y_test \n ",y_test)

#-----------------------------------------------------------------
import numpy as np
from sklearn.model_selection import StratifiedKFold

#split data into parts balanced for Y (output)  , classification بيعمل توزيع متساوي بالنسبه للاوت بوت يستخدم بكثره في ال
X=np.array([[1,2],[3,4],[5,6],[7,8]])
y=np.array([0,0,0,1])

SKF=StratifiedKFold(n_splits=2,shuffle=True)
SKF.get_n_splits(X,y)

print(SKF)

for train_index,test_index in SKF.split(X,y):
    print("Train: " ,train_index, "Test: ",test_index)
    X_train,X_test=X[train_index],X[test_index]
    y_train, y_test = Y[train_index],Y[test_index]
    print("X_Train \n",X_train)
    print("X_Test \n",X_test)
    print("Y_Train \n",y_train)
    print("Y_test \n ",y_test)