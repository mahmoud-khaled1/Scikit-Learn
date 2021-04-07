#Naive Bayes used in Classification type :فكره احتماليه بايز تجعل نسبه احتماليه اي شيي معتمده بشكل او باخر علي النسب السابقه لها
# 1-GaussianNB :تستخدم حينما تكون الفيتشرز موزعة التوزيع الطبيعي (جوسيان) مثل بيانات أيريس
# 2-MultinomialNB:تستخدم مع البيانات المنفصلة, مثلا لو كان لدينا تقييم الافلام يكون ارقام 1 او 2 او 3
# 3-BernoulliNB:تستخدم مع التصنيف الثنائي , صفر او واحد , صحيح او خطا , مريض او غير مريض

#1-GaussianNB 
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import  seaborn as sns
import matplotlib.pyplot as plt

#Load DataSets
Cancer=load_breast_cancer()
X=Cancer.data
Y=Cancer.target

#Splitting DataSets
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44,shuffle=True)

#Applying algorithm

#priors:   القيم المفترضة السابقة
GaussianNBModel=GaussianNB()
GaussianNBModel.fit(X_train,y_train)


#Print Score of algorithm
print("Train Score :",GaussianNBModel.score(X_train,y_train))
print("test Score :",GaussianNBModel.score(X_test,y_test))

#Predected
Y_pre=GaussianNBModel.predict(X_test)
print(list(Y_pre[:20]))
print(list(y_test[:20]))

#confusion_matrix
CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)
#To Visualization it
sns.heatmap(CM,center=True)
plt.show()
#------------------------------------------------------
#2-MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import  seaborn as sns
import matplotlib.pyplot as plt


#Load DataSets
Cancer=load_breast_cancer()
X=Cancer.data
Y=Cancer.target

#Splitting DataSets
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44,shuffle=True)

#Applying algorithm

#alpha : معامل التنعيم
MultinomialNBModel=MultinomialNB(alpha=1.0)
MultinomialNBModel.fit(X_train,y_train)


#Print Score of algorithm
print("Train Score :",MultinomialNBModel.score(X_train,y_train))
print("test Score :",MultinomialNBModel.score(X_test,y_test))

#Predected
Y_pre=MultinomialNBModel.predict(X_test)
print(list(Y_pre[:20]))
print(list(y_test[:20]))

#confusion_matrix
CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)
#To Visualization it
sns.heatmap(CM,center=True)
plt.show()

#----------------------------------------------------------

#1-BernoulliNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import  seaborn as sns
import matplotlib.pyplot as plt

#Load DataSets
Cancer=load_breast_cancer()
X=Cancer.data
Y=Cancer.target

#Splitting DataSets
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44,shuffle=True)

#Applying algorithm

BernoulliNBModel=BernoulliNB(alpha=1.0,binarize=1)
BernoulliNBModel.fit(X_train,y_train)


#Print Score of algorithm
print("Train Score :",BernoulliNBModel.score(X_train,y_train))
print("test Score :",BernoulliNBModel.score(X_test,y_test))

#Predected
Y_pre=BernoulliNBModel.predict(X_test)
print(list(Y_pre[:20]))
print(list(y_test[:20]))

#confusion_matrix
CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)
#To Visualization it
sns.heatmap(CM,center=True)
plt.show()