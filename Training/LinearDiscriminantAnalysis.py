#LinearDiscriminantAnalysis:  classification بس دا بيتعامل مع التعلم باشراف يستخدم في ال  PCA  هو زي ال
#فكرنه بيعمل خط بزاويه معينه يفصل البيانات ولكن الخط دا مش بيفصل التقط عن بعضها زي ال Classification
# فالخط دا المفروض هيتراكم عند مكان فيه شويه بيانات ومكان تاني شويه تانين فكل ما تكون المسافه بين الجروبين كبيره الميو كل ما كان الاجورزم افصل واحسن

#LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

LinearDiscriminantAnalysisModel=LinearDiscriminantAnalysis(n_components=1,solver='svd')
LinearDiscriminantAnalysisModel.fit(X_train,y_train)

#Print Score of algorithm
print("Train Score :",LinearDiscriminantAnalysisModel.score(X_train,y_train))
print("test Score :",LinearDiscriminantAnalysisModel.score(X_test,y_test))

#Predected
Y_pre=LinearDiscriminantAnalysisModel.predict(X_test)
print(list(Y_pre[:20]))
print(list(y_test[:20]))

#confusion_matrix
CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)
#To Visualization it
sns.heatmap(CM,center=True)
plt.show()
#----------------------------------------------------

#Quadratic Discriminant Analysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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

QuadraticDiscriminantAnaModel=QuadraticDiscriminantAnalysis()
QuadraticDiscriminantAnaModel.fit(X_train,y_train)

#Print Score of algorithm
print("Train Score :",QuadraticDiscriminantAnaModel.score(X_train,y_train))
print("test Score :",QuadraticDiscriminantAnaModel.score(X_test,y_test))

#Predected
Y_pre=QuadraticDiscriminantAnaModel.predict(X_test)
print(list(Y_pre[:20]))
print(list(y_test[:20]))

#confusion_matrix
CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)
#To Visualization it
sns.heatmap(CM,center=True)
plt.show()