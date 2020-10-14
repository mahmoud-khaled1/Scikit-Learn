#Classification Algorithms
#1-LogisticRegression


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

# Attach DataSet
DataSets=load_breast_cancer()

X=DataSets.data
Y=DataSets.target

#Splitting data

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=44,shuffle=True,test_size=0.3)

#Applying algorithm

#Solver = 1-liblinear-->small data 2-sag-->big data 3-saga-->big data
#C Regularization
#tol مقدار السماحية المسموح بها في الخطأ
Logistic_Regression_model=LogisticRegression(penalty='l2',solver='sag',C=0.1,random_state=44,n_jobs=-1,max_iter=1000)
Logistic_Regression_model.fit(X_train,y_train)

#Calculation Details

#Score  value is between [0,1] when is near to 1 this is mean Over Fitting ,when is near to 0 mean Under Fitting
# So we Should make it balanced Score .
print("Train Score :",Logistic_Regression_model.score(X_train,y_train))
print("test Score :",Logistic_Regression_model.score(X_test,y_test))
print("Classes are :",Logistic_Regression_model.classes_)
#We have max_iter=10000 but here will found max_itr =4059 iteration that mean that algorithm after 4059 iteration not necessary
print("Max Iteration :",Logistic_Regression_model.n_iter_)
print("*"*100)

#Make Predicted
y_pre=Logistic_Regression_model.predict(X_test)
#Probability of every item in each classifer
y_pre_pro=Logistic_Regression_model.predict_proba(X_test)

print(y_pre_pro[:15])
print(y_test[:15])
print(y_pre[:15])

#Calculted Coufusion matrix
#[TP     FP
#[FN     TN]
##We should make TP and TN biggest and smallest FP and FN
CM=confusion_matrix(y_test,y_pre)
print("Confusion Matrix :",CM)

#Draw Coufusion matrix
sns.heatmap(CM,center=True)
plt.show()

#Calculating Accuracy Score :((TP+TN)/float(TP+TN+FP+FN))

# If We make normalize= False then will calc sum of  (TP+TN) , and if make it True Then will display as precision ((TP+TN)/float(TP+TN+FP+FN))
ACC_Score=accuracy_score(y_test,y_pre,normalize=False)
print("Accuracy Score :",ACC_Score)

#Calc F1_Score = (2*(Precision * recall)(precision+recall))
#F1 Score Calculation
#The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches
# its best value at 1 and worst score at 0.
# The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
#F1 Score=2*(precision * recall)/(precision+recall)

F1_Score =f1_score(y_test,y_pre)

print("F1_Score :",F1_Score)


# Precision =(TP)/(TP+FP)
from sklearn.metrics import precision_score
Precision=precision_score(y_true=y_test,y_pred=y_pre,labels=None,average='binary')
print("Precision :",Precision)


# recall_score =(TP)/(TP+FP)
from sklearn.metrics import recall_score
recall=recall_score(y_true=y_test,y_pred=y_pre,labels=None,average='binary')
print("recall :",recall)


#Classsification Report
Class_Report=classification_report(y_test,y_pre,labels=None)

print("Classification Report :")
print(Class_Report)

# نسيه عدم التطابق بين القيم الحقيقيه والمتوقعه
from sklearn.metrics import zero_one_loss

Zero_ones=zero_one_loss(y_test,y_pre)
print(Zero_ones)
