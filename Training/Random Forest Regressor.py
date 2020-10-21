#1-Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


Boston=load_boston()
X=Boston.data
Y=Boston.target

print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True,random_state=44,test_size=0.23)


#n_estimators :عدد الشجر المستخدم
#max_depth: العمق المطلوب
#min_samples_split : الحد الادني من التقسيمات المسموح بها

#for i in range(100,1000,100):
RandomForestRegressorModel=RandomForestRegressor(n_estimators=100,max_depth=7,random_state=44)
RandomForestRegressorModel.fit(X_train,y_train)

print("Train Score :",RandomForestRegressorModel.score(X_train,y_train))
print("test Score :",RandomForestRegressorModel.score(X_test,y_test))
print("no od features  :",RandomForestRegressorModel.n_features_)


#Predected
Y_pre=RandomForestRegressorModel.predict(X_test)
print(list(y_test[:10]))
print(list(Y_pre[:10]))


MAE_Value=mean_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAE_Value)

MAESqu_Value=mean_squared_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(MAESqu_Value)

median_value =median_absolute_error(y_true=y_test,y_pred=Y_pre,multioutput='uniform_average')
print(median_value)

