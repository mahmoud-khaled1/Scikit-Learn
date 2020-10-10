import pandas as pd

dataset=pd.read_csv('D:\\ML\\SKLearn Library\\Slides && Data\Data\\2.9 Ensemble Reg\\houses.csv')

#print all dataset in houses file
print(dataset)

#to avoid  specific feature in this data we use function drop in pandas
newdataset=dataset.drop['bedrooms']

#print dataset after avoid feature bedrooms
print(newdataset)

#------------------------------
#to make algorithm avoid useless feature automatic from data we can do that with module feature_selection
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2,f_classif
from sklearn.datasets import load_breast_cancer

BreastData=load_breast_cancer()

X=BreastData.data
Y=BreastData.target

print("X Shape",X.shape)
print("Y Shape",Y.shape)

#Now we will select  important 20% from feature of data
Feature_Selection=SelectPercentile(score_func=chi2,percentile=20)
Xx=Feature_Selection.fit_transform(X,Y)

print("X Shape after select  20% feature from them  :",Xx.shape)

print("Selected Feature Are : ",Feature_Selection.get_support())
#output True and False True that feature is selected and False that feature not selected

#-------------------------------------------
#Generic Univariate select  =>>this method of selected data i gave it specific number of feature

from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2,f_classif
from sklearn.datasets import load_breast_cancer
CancerData=load_breast_cancer()
X=CancerData.data
Y=CancerData.target
print('X Shape :',X.shape)
print("Y Shape :",Y.shape)

# Get best 3 Feature in data Feature
Feature_Selected=GenericUnivariateSelect(score_func=chi2,mode='k_best',param=3)
X=Feature_Selected.fit_transform(X,Y)
print('X Shape :',X.shape)
print("Y Shape :",Y.shape)

#--------------------------------------
#Select From model Just gave data to specific model like SVM, Random Forest ... to select best feature
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

Cancer=load_breast_cancer()
X=Cancer.data
Y=Cancer.target
print("X Shape",X.shape)
model=RandomForestClassifier(n_estimators=20)
Feature_Selected= SelectFromModel(estimator=model)
X=Feature_Selected.fit_transform(X,Y)
print(Feature_Selected.get_support())
print("X Shape",X.shape)

