#1- SVR for Regression

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#Data Sets
Load_Booston_model=load_boston()
X=Load_Booston_model.data
Y=Load_Booston_model.target

#Splitting Data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,shuffle=True,random_state=44)

#Apply algorithm SVR

#Kernal : (rbf , linear , poly , sigmoid , precomputed)
#degree :set degree of equation if i select kernal = poly
#epsilon :	قيمة ابسلون المستخدمة
SVR_Model=SVR(kernel='linear',C=0.1,max_iter=100000,epsilon=0.1)
SVR_Model.fit(X_train,y_train)

#Print Details
print("Train Score :",SVR_Model.score(X_train,y_train))
print("Test Score :",SVR_Model.score(X_test,y_test))
print("No of Iteration :",SVR_Model.max_iter)

#make Predicted

Y_pre=SVR_Model.predict(X_test)

print(list(Y_pre[:10]))
print(list(y_test[:10]))

#----------------------------------------
#Another Example
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split

#Attach DataSets
import pandas as pd
DataSets=pd.read_csv('D:\\ML\\SKLearn Library\\Slides && Data\Data\\2.4 SVR\\Earthquakes.csv')


#Data Cleaning
from sklearn.impute import SimpleImputer
import numpy as np
#missing_value : is a Value you want to avoid it and want to replace it with anything
# like = np.nan null data,=0 data with value 0 .....
# Strategy what is method you want  your algorithm go on
# to replace this missing_value (meanالمتوسط الحسابي =مجموعهم علي عددهم ,median يرتب الارقام وياخد اللي في النص ,most_frequency الاكثر تكرار في الارقام,constant)
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp=imp.fit(DataSets)
modifiedData=imp.transform(DataSets)

#Data Scaling
from sklearn.preprocessing import StandardScaler
#Copy= True Just told algorithm don't change the data
Data_Scaling =StandardScaler(copy=True,with_mean=True,with_std=True)#.fit_transform(data)
Scaling_data=Data_Scaling.fit_transform(modifiedData)

#Data Splitting
X=modifiedData[:,:-1]
Y=modifiedData[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,shuffle=True)

#Applying Algorithm
#Kernal : (rbf , linear , poly , sigmoid , precomputed)
#degree :set degree of equation if i select kernal = poly
#epsilon :	قيمة ابسلون المستخدمة
SVR_Model=SVR(kernel='rbf',max_iter=10000,epsilon=0.1)
SVR_Model.fit(X_train,y_train)

#Print Details
print("Train Score :",SVR_Model.score(X_train,y_train))
print("Test Score :",SVR_Model.score(X_test,y_test))
print("No of Iteration :",SVR_Model.max_iter)

#make Predicted
Y_pre=SVR_Model.predict(X_test)
print(list(Y_pre[:10]))
print(list(y_test[:10]))



