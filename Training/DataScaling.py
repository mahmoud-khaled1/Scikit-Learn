#Data Scaling : is the simplest method and consists in rescaling the range of features to scale the range in [0, 1],[-1,1]
#we used Data Scaling to make processing faster


#1-Standardization :'X' of every column = (X - ((μ)mean of column)) /(standard division)
#SD=sqrt((∑∣x−μ∣^2)/N)
# where μ is mean the data and N is number of data

from sklearn.preprocessing import StandardScaler

data=[[5421,0],
      [0,1000],
      [1,154],
      [18754,1]]
#Copy= True Just told algorithm don't change the data
Data_Scaling =StandardScaler(copy=True,with_mean=True,with_std=True)#.fit_transform(data)
Scaling_data=Data_Scaling.fit_transform(data)
print(Scaling_data)

#--------------------------------------------
#Normalization : 'X' of every column = (X - ((μ)mean of column)) / (max(X) - min(X))
from sklearn.preprocessing import MinMaxScaler
data=[[5421,0],
      [0,1000],
      [1,154],
      [18754,1]]
Min_Max_Sacler=MinMaxScaler(feature_range=(0,1),copy=True)
data_afer_Scaling=Min_Max_Sacler.fit_transform(data)
print(data_afer_Scaling)

#--------------------------------------------
#Normalizer : Take Every column Alone

from sklearn.preprocessing import Normalizer

data=[[4, 2],
      [3, 9],
      [1, 5],
      [4, 1]]

#norm can be (l1,l2,max)
#we used l1 to make summation of every row is max value ,then we will calc precent of every X in row  of max value
#we used l2 to make sqrt of summation of every  row is max value ,then we will calc precent of every X of max value
# we used max to make max value of every row  is max value ,then we will calc precent of every X of max value
Norm=Normalizer(norm='l1')
data_norm=Norm.fit_transform(data)
print(data_norm)

#--------------------------------------------
#MaxAbsScaler : look like Normalizer but here we calc of every column not row like Normalizer

from sklearn.preprocessing import MaxAbsScaler
data=[[4, 2],
      [3, 9],
      [1, 5],
      [4, 1]]

#we used l1 to make summation of every column is max value ,then we will calc precent of every X in column  of max value
#we used l2 to make sqrt of summation of every  column is max value ,then we will calc precent of every X of max value
# we used max to make max value of every column  is max value ,then we will calc precent of every X of max value
Norm=MaxAbsScaler(copy=True)
data_norm=Norm.fit_transform(data)
print(data_norm)

#--------------------------------------------
#FunctionTransformer : make Scaling of my data but by passing function to her to calc it

from sklearn.preprocessing import FunctionTransformer
data=[[4, 2],
      [3, 9],
      [1, 5],
      [4, 1]]

#Sunc=lamda or specific function I created it .
Fun_Trans=FunctionTransformer(func=lambda x:x**0.1,validate=True)
data_afer_Scal=Fun_Trans.fit_transform(data)
print(data_afer_Scal)

#--------------------------------------------
#Binarizer : make evey data in evert column 0,1 only
from sklearn.preprocessing import Binarizer

data=[[4, 2],
      [3, 9],
      [1, 5],
      [4, 1]]
#Every number in data greater than 5.0 will be 1 and lesser than 5.0 will be 0
Binar=Binarizer(threshold=5.0)
data_afer_Scal=Binar.fit_transform(data)
print(data_afer_Scal)

#--------------------------------------------

#Polynomial Features : is a method to increase number of Feature in my data by multiply all column in polynomial method
# used it when my algorithm have under fitting problem

from sklearn.preprocessing import  PolynomialFeatures
data=[[4, 2],
      [3, 9],
      [1, 5],
      [4, 1]]
# degree is degree of my polynomial function if my degree 2 have 2 feature then will be 1,a,b,a^2,ab,b^2 in column
#include_bias True if i want to add column 1 in my new Feature
Poly=PolynomialFeatures(degree=2,interaction_only=False,include_bias=True)
Data_scal=Poly.fit_transform(data)
print(Data_scal)

