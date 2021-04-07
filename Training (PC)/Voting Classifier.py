#•	خوارزم التصنيف بالتصويت بين عدد من الخوارزميات المختلفة
#take all classification algorithms and train them then voting all algorithm to classifier what i want .
#Before use voting algorithm you should declare all classifier algorithms that voting algorithm wil use them

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import  seaborn as sns
import matplotlib.pyplot as plt


#Load DataSet
Cancer =load_breast_cancer()
X=Cancer.data
Y=Cancer.target

#Split DataSet
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=44,test_size=0.33,shuffle=True)


#Load Models for voting algorithm
DT_Model=DecisionTreeClassifier(criterion="entropy",random_state=44,max_depth=3)
LDA_Model=LinearDiscriminantAnalysis(n_components=1,solver='svd')
SGD_Model=SGDClassifier(loss='log',penalty='l2',max_iter=10000,tol=1e-5)

#Load Voting Algorithm
#Weights is weight of every algorithm to make voting  sum of them = 1
#estimators take array of Tuple for algorithms .
VotingClassifierModel=VotingClassifier(estimators=[("DecisionTree",DT_Model),("LDA_Model",LDA_Model),("SGD_Model",SGD_Model)],
                                        voting='hard',weights=None)
VotingClassifierModel.fit(X_train,y_train)


#Print Score of algorithm
print("Train Score :",VotingClassifierModel.score(X_train,y_train))
print("test Score :",VotingClassifierModel.score(X_test,y_test))

#Predected
Y_pre=VotingClassifierModel.predict(X_test)

print(list(Y_pre[:20]))
print(list(y_test[:20]))


#confusion_matrix


CM=confusion_matrix(y_true=y_test,y_pred=Y_pre)
print(CM)
#To Visualization it
sns.heatmap(CM,center=True)
plt.show()
























