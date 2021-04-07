from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

#Data Sets
Breast_Cancer=load_breast_cancer()
X=Breast_Cancer.data
Y=Breast_Cancer.target

#Split Data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=44,shuffle=True)

#Applying algorithm
SVC_model=SVC(C=1,kernel='linear',gamma='auto',max_iter=10000,random_state=44)
SVC_model.fit(X_train,y_train)


#Print Details
print("Train Score :",SVC_model.score(X_train,y_train))
print("Test Score :",SVC_model.score(X_test,y_test))
print("No of Iteration :",SVC_model.max_iter)

#make Predicted

Y_pre=SVC_model.predict(X_test)

print(list(Y_pre[:10]))
print(list(y_test[:10]))


from sklearn.metrics import confusion_matrix
CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)


