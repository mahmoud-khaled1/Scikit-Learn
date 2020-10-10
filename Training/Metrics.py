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
#The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches
# its best value at 1 and worst score at 0.
# The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
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
from sklearn.metrics import precision_recall_fscore_support
# this class calc precision and recall and f1 score and support at the same time Return array of them
y_true=[1,0,1,1,0,0,1]
y_pre= [1,0,0,1,0,1,1]

Pre_Recall_F1Score=precision_recall_fscore_support(y_true=y_true,y_pred=y_pre,average='micro')

print(Pre_Recall_F1Score)

#------------------------------------------------
from sklearn.metrics import precision_recall_curve
y_true=[1,0,1,1,0,0,1]
y_pre= [1,0,0,1,0,1,1]

precsion,recall,thresholds=precision_recall_curve(y_pre,y_true)

print(precsion)
print(recall)
print(thresholds)
#-------------------------------------------------
#classification Report
#وهي تقوم بحساب كلا من : precision , recall , f1score , support  لكل قيمة من القيم , سواء ارقام او نصوص , كما تقوم بإظهار المتوسطات بأنواعها macro , micro , support

from sklearn.metrics import classification_report
y_true=[1,0,1,1,0,0,1]
y_pre= [1,0,0,1,0,1,1]

CR=classification_report(y_true=y_true,y_pred=y_pre)

print(CR)
#-------------------------------------------------
#اداة الـ ROC و هي اختصار Receiver operating characteristic , وهي فقط تستخدم مع التصنيف الثنائي binary classification

#هي اداة لتحديد القيمة المناسبة للـ sensitivity & specificity  , واختيار الـ threshold  المناسبة , مع ملاحظة اننا نعطيها y_pred_prob  فهي تاخذ احتمالية و ليست قيم متوقعة

#تعطي 3 قيم , fpr  و هي false positive rate  و تساوي 1- specificity و tpr  و هي true positive rate و تساوي sensitivity و قيمة الثريشهولد المناسبة

from sklearn.metrics import roc_curve

#Calculating Receiver Operating Characteristic :
#roc_curve(y_true, y_score, pos_label=None, sample_weight=None,drop_intermediate=True)

y_test=[1,0,1,1,0,0,1]
y_pre= [1,0,0,1,0,1,1]
fprValue, tprValue, thresholdsValue = roc_curve(y_test,y_pre)
print('fpr Value  : ', fprValue)
print('tpr Value  : ', tprValue)
print('thresholds Value  : ', thresholdsValue)
#-------------------------------------------------

#أداة AUC  و هي اختصار area under curve
#و هي التي تقوم بحساب المساحة تحت المنحني السابق شرحه , ولاحظ ان كلما زادت المساحة تحت المنحني كلما دل هذا علي دقة الخوارزم , وذلك لانه يتيح قيم عالية الـ sensitivity & specificity   معا
#Calculating Area Under the Curve :
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fprValue2, tprValue2, thresholdsValue2 = roc_curve(y_test,y_pre)
AUCValue = auc(fprValue2, tprValue2)
#print('AUC Value  : ', AUCValue)

import numpy as np
from sklearn import metrics
y =      np.array([1    , 1     , 2     , 2])
scores = np.array([0.1  , 0.4   ,   0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

metrics.auc(fpr, tpr)
#---------------------------------------------
#و هذا الأمر يجمع الأمرين السابقين معا , اذ اننا نقوم بحساب auc  مباشرة من القيم دون تطبيق  roc   اولا
#Import Libraries
from sklearn.metrics import roc_auc_score
#----------------------------------------------------
#Calculating ROC AUC Score:
#roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)

ROCAUCScore = roc_auc_score(y_test,y_pre, average='micro') #it can be : macro,weighted,samples
#print('ROCAUC Score : ', ROCAUCScore)

import numpy as np
from sklearn import metrics
y =      np.array([1    , 1     , 2     , 2])
scores = np.array([0.1  , 0.4   ,   0.35, 0.8])
metrics.roc_auc_score(y, scores)
#---------------------------------------------
#و هي تقوم بحساب عدد مرات اللا تطابق  . .

# Import Libraries
from sklearn.metrics import zero_one_loss
# Calculating Zero One Loss:
# zero_one_loss(y_true, y_pred, normalize = True, sample_weight = None)
ZeroOneLossValue = zero_one_loss(y_test, y_pre, normalize=False)
# print('Zero One Loss Value : ', ZeroOneLossValue )
from sklearn.metrics import zero_one_loss

y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
zero_one_loss(y_true, y_pred)
zero_one_loss(y_true, y_pred, normalize=False)
