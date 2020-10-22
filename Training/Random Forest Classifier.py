#1-Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

Cancer=load_breast_cancer()
X=Cancer.data
Y=Cancer.target

print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True,random_state=44,test_size=0.23)


RandomForestClassifierModel=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=3,random_state=44)
RandomForestClassifierModel.fit(X_train,y_train)


print("Train Score :",RandomForestClassifierModel.score(X_train,y_train))
print("test Score :",RandomForestClassifierModel.score(X_test,y_test))
print("no of features  :",RandomForestClassifierModel.n_features_)
print(" importace  features  :",RandomForestClassifierModel.feature_importances_)

#Predected
Y_pre=RandomForestClassifierModel.predict(X_test)
print(list(y_test[:10]))
print(list(Y_pre[:10]))



from sklearn.metrics import confusion_matrix

CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)

from sklearn.metrics import accuracy_score
ACCSCORE=accuracy_score(y_true=y_test,y_pred=Y_pre,normalize=True)
print("Accuracy",ACCSCORE*100,'%')

#-----------------------------------------------------------------------------

#2-Gradient Boosting Classifier
##تعتمد علي استخدام عدد كبير من الخوارزميات الغير قوية معا , لعمل نتيجة قوية و ايجابية  مشابهة لفكرة الغابة العشوائية , من قرارات عديدة غير قوية تاتي بنتيجة قوي



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

Cancer=load_breast_cancer()
X=Cancer.data
Y=Cancer.target

print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True,random_state=44,test_size=0.23)



GBC=GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=44)
GBC.fit(X_train,y_train)


print("Train Score :",GBC.score(X_train,y_train))
print("test Score :",GBC.score(X_test,y_test))
print("no of features  :",GBC.n_features_)
print(" importace  features  :",GBC.feature_importances_)

#Predected
Y_pre=GBC.predict(X_test)
print(list(y_test[:10]))
print(list(Y_pre[:10]))



from sklearn.metrics import confusion_matrix

CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)

from sklearn.metrics import accuracy_score
ACCSCORE=accuracy_score(y_true=y_test,y_pred=Y_pre,normalize=True)
print("Accuracy",ACCSCORE*100,'%')













