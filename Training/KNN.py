#KNN : use in Regression , Classification , unsupervised

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


#Load dataSets
Boston =load_boston()
X=Boston.data
Y=Boston.target

#Split DataSet
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44,shuffle=True)

#Appltying algorithm
KNeighborsRegressorModel=KNeighborsRegressor(n_neighbors=3,weights='uniform',algorithm='auto')
KNeighborsRegressorModel.fit(X_train,y_train)


#Print Score of algorithm
print("Train Score :",KNeighborsRegressorModel.score(X_train,y_train))
print("test Score :",KNeighborsRegressorModel.score(X_test,y_test))

#Predected
Y_pre=KNeighborsRegressorModel.predict(X_test)

print(list(Y_pre[:20]))
print(list(y_test[:20]))


MAE_Value=mean_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAE_Value)

MAESqu_Value=mean_squared_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAESqu_Value)

median_value =median_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(median_value)
