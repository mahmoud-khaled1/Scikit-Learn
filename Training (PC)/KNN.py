#KNN : use in Regression , Classification , unsupervised
#1-SuperVised
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


#Load dataSets
IRIS =load_iris()
X=IRIS.data
Y=IRIS.target

#Split DataSet
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44,shuffle=True)

#Appltying algorithm
import numpy as np

Scores=[]
K_ranage=range(1,20)
from sklearn import metrics
for i in K_ranage:
    KNeighborsClassifierModel=KNeighborsClassifier(n_neighbors=i,weights='uniform',algorithm='auto')
    KNeighborsClassifierModel.fit(X_train,y_train)
    Y_pre=KNeighborsClassifierModel.predict(X_test)
    Scores.append(metrics.accuracy_score(y_test,Y_pre))


import matplotlib.pyplot as plt

plt.plot(K_ranage,Scores)
plt.xlabel("Value for K in KNN")
plt.ylabel("Testing Accuracy")
plt.show()

#Print Score of algorithm
#print("Train Score :",KNeighborsRegressorModel.score(X_train,y_train))
#print("test Score :",KNeighborsRegressorModel.score(X_test,y_test))

#Predected
#Y_pre=KNeighborsRegressorModel.predict(X_test)

print(list(Y_pre[:20]))
print(list(y_test[:20]))


MAE_Value=mean_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAE_Value)

MAESqu_Value=mean_squared_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAESqu_Value)

median_value =median_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(median_value)

#-------------------------------------------------------------------------
#1-UnSuperVised

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import pandas as pd

#Load DataSets
DataSets=pd.read_csv(r'D:\\ML\\SKLearn Library\\Slides && Data\\Data\\2.7 PCA\\data.csv')
X=DataSets
X_train=X.iloc[:1000,:]
X_test=X.iloc[100:,:]
#print(X.shape)

#Applying algorithm
NearestNeighborsModel=NearestNeighbors(n_neighbors=5,radius=1.0,algorithm='auto')
NearestNeighborsModel.fit(X_train)

#Show Details
print("NearestNeighborsModel Train KNeighbors",NearestNeighborsModel.kneighbors(X_train[:5]))
print("NearestNeighborsModel Train radius KNeighbors",NearestNeighborsModel.radius_neighbors(X_train[:1]))

print("NearestNeighborsModel Test KNeighbors",NearestNeighborsModel.kneighbors(X_test[:5]))
print("NearestNeighborsModel Test radius KNeighbors",NearestNeighborsModel.radius_neighbors(X_test[:1]))