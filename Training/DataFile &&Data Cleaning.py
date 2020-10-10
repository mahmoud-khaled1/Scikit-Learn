import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Data File with SkLearn


from sklearn.datasets import load_iris
#Iris Data
IrisData=load_iris()
x=IrisData.data
y=IrisData.target

print(x[:10])
print(x.shape)
print(IrisData.feature_names)
print(y)


#-------------------------------------------------------

#Data Digit

from sklearn.datasets import load_digits

DigitData=load_digits()
x1=DigitData.data
y1=DigitData.target

print(x1[:10])
print("***************")
print(y1)

plt.gray()

for i in range(0,11):
    print("image of number:",i)
    plt.matshow(DigitData.images[i])
    plt.show()

#-----------------------------------------------------

# Boston data is data of price of House in Boston country in America

from sklearn.datasets import load_boston

BostonData=load_boston()

X=BostonData.data

Y=BostonData.target

feature=BostonData.feature_names

print("X" , X.shape)
print(X[:10])

print("Y",Y.shape)
print(Y[10])

print("Feature:")

print(feature)


#----------------------------------------------------
#Multie Classification of drinks Wines

from sklearn.datasets import load_wine
#save in object
WineData=load_wine()
# rows
X=WineData.data
#Y result
Y=WineData.target

feature=WineData.feature_names
print(X[:1])
print('*'*50)
print(Y[:100])
print('*'*50)
print(feature)

#----------------------------------------------
#Breast Cancer Data سرطان الثدي
#Data of classification of breast cancer if this person have or not this cancer

from sklearn.datasets import load_breast_cancer

breastData=load_breast_cancer()
X=breastData.data
Y=breastData.target
feature=breastData.feature_names

print(feature)
print("X Shape")
print(X.shape)
print(X[:1])
print(Y[:200])

#----------------------------------------------
# Data for diabetes Regression  مرض السكر

from sklearn.datasets import load_diabetes

diabetes_Data=load_diabetes()

X=diabetes_Data.data
Y=diabetes_Data.target
feature=diabetes_Data.feature_names

print(X[:1])
print(Y[:10])
print(feature)


#----------------------------------------
#make Regression  sample Regression

from sklearn.datasets import make_regression

X,Y=make_regression(n_samples=1000,n_features=50,shuffle=True)

print(X.shape)
print(Y.shape)
print(X)
print(Y)

#----------------------------------------
#make Classification  sample Classification

from sklearn.datasets import make_classification

X,Y=make_classification(n_samples=1000,n_features=50,shuffle=True)
print(X.shape)
print(Y.shape)
print(X[:10])
print(Y[:10])

#----------------------------------------
#make image  sample Image import some image from class load_sample image

from sklearn.datasets import load_sample_image
china=load_sample_image('flower.jpg')

import matplotlib.pyplot as plt

plt.imshow(china)
plt.show()


#-------------------------------------------------------------------------------------------------------------------------
#Data Cleaning

from sklearn.impute import SimpleImputer
import numpy as np
#missing_value : is a Value you want to avoid it and want to replace it with anything
# like = np.nan null data,=0 data with value 0 .....
# Strategy what is method you want  your algorithm go on
# to replace this missing_value (meanالمتوسط الحسابي =مجموعهم علي عددهم ,median يرتب الارقام وياخد اللي في النص ,most_frequency الاكثر تكرار في الارقام,constant)
imp=SimpleImputer(missing_values=0,strategy='mean')
data=[[1,2,0],
      [3,0,1],
      [5,0,0],
      [0,4,6],
      [5,0,0]]

imp=imp.fit(data)

modifiedData=imp.transform(data)

print(modifiedData)

