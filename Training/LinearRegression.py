# ' Leaner Regression algorithm '
#Step of any algorithm in machine learning :
   #1-Data File and attach data
   #2-Data Cleaning
   #3-Feature selection
   #4-Data Scaling
   #5-Data Splitting
   #6-Choice best algorithm (Regression-Classification-SVM-.......)
#-------------------------------------------------------------------------------
# 1-Linear Regression class is model but without Regularization (most common algorithm of them)

#import Libraries and modules
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

#import DataSet
Boston_Data=load_boston()
X=Boston_Data.data
Y=Boston_Data.target

#Split Data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=44,shuffle=True)

#Applying Algorithm

# fit_intercept : هل نريد حساب نقطه التقاطع مع محور اكس ام لا
# normalize : make normalization to data or not
#copy_x : if you want to just copy only the data without any change of them
#n_jobs : is  specify speed of operation in your processor if =0 or none will be normal ,-1 is faster speed of processor
# and the more than 1 the more than speed
Linear_Reg_model=LinearRegression(fit_intercept=True,normalize=True ,copy_X=True,n_jobs=-1)
Linear_Reg_model.fit(X_train,y_train)


#Show Details
print("Linear Regression Train Score :",Linear_Reg_model.score(X_train,y_train))
print("Linear Regression test Score :",Linear_Reg_model.score(X_test,y_test))
print("Linear Regression Coef Score :",Linear_Reg_model.coef_)
print("Linear Regression intercept Score :",Linear_Reg_model.intercept_)

#Calculating Predication
Y_pred=Linear_Reg_model.predict(X_test)
print("Predicted value for linear Regression :",Y_pred[:10])
print("Predicted value for linear Regression :",y_test[:10])

#Calculating mean_absolute_error
MeanVal=mean_absolute_error(y_true=y_test,y_pred=Y_pred,multioutput='uniform_average')
print("mean absolute error :",MeanVal)

#Calculating mean_squared_error
MeanSquVal=mean_squared_error(y_true=y_test,y_pred=Y_pred,multioutput='uniform_average')
print("mean absolute error :",MeanSquVal)

#Calculating median_absolute_error
MedVal=median_absolute_error(y_true=y_test,y_pred=Y_pred,multioutput='uniform_average')
print("mean absolute error :",MedVal)

#-----------------------------------------------
#Another Example  متغر واخد

#import Libraries and modules
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##import DataSet
DataSet=pd.read_csv('D:\\ML\\SKLearn Library\\Slides && Data\Data\\2.1 Linear Regression\\satf.csv')

#print(DataSet[:5])

#select first column as X and last column as Y from DataSet
X=DataSet.iloc[:,:1]
Y=DataSet.iloc[:,-1]

#Split Data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=44,shuffle=True)


#Applying Algorithm
Linear_Reg_model=LinearRegression(fit_intercept=True,normalize=True ,copy_X=True,n_jobs=-1)
Linear_Reg_model.fit(X_train,y_train)


#Show Details
#print("Linear Regression Train Score :",Linear_Reg_model.score(X_train,y_train))
#print("Linear Regression test Score :",Linear_Reg_model.score(X_test,y_test))

#Calculating Predication

Y_pred=regressor.predict(X_test)

print(y_test[:5])
print(Y_pred[:5])


#Calculating mean_absolute_error
MeanVal=mean_absolute_error(y_true=y_test,y_pred=Y_pred,multioutput='uniform_average')
print("mean absolute error :",MeanVal)

#Calculating mean_squared_error
MeanSquVal=mean_squared_error(y_true=y_test,y_pred=Y_pred,multioutput='uniform_average')
print("mean absolute error :",MeanSquVal)

#Calculating median_absolute_error
MedVal=median_absolute_error(y_true=y_test,y_pred=Y_pred,multioutput='uniform_average')
print("mean absolute error :",MedVal)

#visualization Training sets

plt.scatter(X_train,y_train,color='red')
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train.regressor.predict(X_train),color='Blue')
plt.title("Sat Degree")
plt.xlabel("High GPA")
plt.ylabel('Unive_GPA')
plt.show()