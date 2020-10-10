# metrics use to calc Error of algorithm in training data == Cost function

# 1-Metrics for Regression

#Cost function of Regression == Mean Absolute Error Sum((Y_of algorithm)-(Y_True))/N

from sklearn.metrics import mean_absolute_error

y_true=[1,2,5,8,8,66,988]
y_pre=[1,2,6,5,7,60,900]

MAE_Value=mean_absolute_error(y_true=y_true,y_pred=y_pre,multioutput='uniform_average')
#y_true are  correct output of data , y_pred are my algorithm output
#multioutput has 2 value 1-uniform_average if output is single value here calc mean value of all row together
# and 2-row_value if has multi value(Two dimensional array) here calc mean value of each row output will be array
print(MAE_Value)
#----------------------------------------------
#Cost function of Regression == Mean squared Error Sum ((Y_of algorithm)-(Y_True))^2/N most common of them

from sklearn.metrics import mean_squared_error

MAESqu_Value=mean_squared_error(y_true=y_true,y_pred=y_pre,multioutput='uniform_average')

print(MAESqu_Value)
#----------------------------------------------
# calc error with median by subtract y_true and y_pred of every row then sort them and choice the median of them
from sklearn.metrics import median_absolute_error

median_value =median_absolute_error(y_true=y_true,y_pred=y_pre,multioutput='uniform_average')

print(median_value)

#-----------------------------------------------------------------------------------------------------------------------------------

#2-Metrics for Classification

#Confusion matrix ==> [TP    FP ]
#                      FN    TN
#We should make TP and TN biggest and smallest FP and FN

import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix

y_true=[1,0,0,1,0,1,1]
y_pre=[0,0,0,1,0,1,0]
CM=confusion_matrix(y_true=y_true,y_pred=y_pre)
print(CM)
#To Visualization it
sb.heatmap(CM,center=True)
plt.show()
#----------------------------------------------
#Accuracy Score =((TP+TN)/float(TP+TN+FP+FN))
from sklearn.metrics import accuracy_score
y_true=[1,0,0,1,0,1,1]
y_pre= [1,0,0,1,0,1,1]

#if we don't put normalize=True  we will calc it as ((TP+TN)/float(TP+TN+FP+FN))
#but if we put it =False will calc number of TP and TN in your data
ACCSCORE=accuracy_score(y_true=y_true,y_pred=y_pre,normalize=True)

print("Accuracy",ACCSCORE)
#----------------------------------------------
#F1 Score Calculation
#F1 Score=2*(precision * recall)/(precision+recall)

from sklearn.metrics import f1_score

y_true=[1,0,0,1,0,0,1]
y_pre= [1,0,0,1,0,1,1]
F1_Score=f1_score(y_true=y_true,y_pred=y_pre,average='binary') # average can be (binary,micro,samples,weighted)

print(F1_Score)

#----------------------------------------------
# Precision =(TP)/(TP+FP)
from sklearn.metrics import precision_score
y_true=[1,0,1,1,0,0,1]
y_pre= [1,0,0,1,0,1,1]

Pre=precision_score(y_true=y_true,y_pred=y_pre,labels=None,average='binary')

print(Pre)

#----------------------------------------------
# Precision =(TP)/(TP+FN)
from sklearn.metrics import recall_score
y_true=[1,0,1,1,0,0,1]
y_pre= [1,0,0,1,0,1,1]

Pre=recall_score(y_true=y_true,y_pred=y_pre,labels=None,average='micro')

print(Pre)
#----------------------------------------------

