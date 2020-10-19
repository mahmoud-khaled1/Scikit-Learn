#1-Decision Tree For Regression

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


Boston=load_boston()
X=Boston.data
Y=Boston.target

#Spllint data
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=44,test_size=0.22,shuffle=True)

#Applying algorithm
Decision_Tree_model=DecisionTreeRegressor(criterion='friedman_mse',splitter='best',max_depth=3,min_samples_split=2,
                                          min_samples_leaf=2,max_features='auto')
Decision_Tree_model.fit(X_train,y_train)

#Show Details
print("Train Score :",Decision_Tree_model.score(X_train,y_train))
print("Test Score :",Decision_Tree_model.score(X_test,y_test))

#Calculation Prediction
Y_pre=Decision_Tree_model.predict(X_test)
print(list(Y_pre[:10]))
print(list(y_test[:10]))

MAE_Value=mean_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAE_Value)

MAESqu_Value=mean_squared_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAESqu_Value)

median_value =median_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(median_value)

#------------------------------------------------------------------
#2-Decision Tree For Classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

breast_cancer=load_breast_cancer()
X=breast_cancer.data
Y=breast_cancer.target


from sklearn.preprocessing import StandardScaler

Data_Scaling =StandardScaler(copy=True,with_mean=True,with_std=True)#.fit_transform(data)
X=Data_Scaling.fit_transform(X)



#Spllint data
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=44,test_size=0.22,shuffle=True)

#Applying algorithm
Decision_Tree_model=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3,min_samples_split=2,
                                          min_samples_leaf=2,max_features='auto')
Decision_Tree_model.fit(X_train,y_train)

#Show Details
print("Train Score :",Decision_Tree_model.score(X_train,y_train))
print("Test Score :",Decision_Tree_model.score(X_test,y_test))
print("Calsses of model ",Decision_Tree_model.classes_)
print("max Feature of model ",Decision_Tree_model.feature_importances_) #important of every Feature
#Calculation Prediction
Y_pre=Decision_Tree_model.predict(X_test)
print(list(Y_pre[:10]))
print(list(y_test[:10]))

from sklearn.metrics import confusion_matrix
CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)