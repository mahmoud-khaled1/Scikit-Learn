#   Natural Language Processing (NLP)
#	تختص بشكل أساسي في التعامل مع النصوص و ما يسمي تحليل الإنطباع Sentiment Analysis
#	     يتم استخدامها عبر اثنين من الموديلوز في sklearn  وهي :
#	     preprocessing.LabelEncoder				لتحويل النصوص في الفيتشرز الي ارقام
#	     preprocessing.OneHotEncoder			لصناعة مصفوفة الواحد من النصوص
#	     feature_extraction.text.CountVectorizer		لقراءة النصوص الطويلة و معالجتها

#1-LabelEncoder : used in ML to convert test in column to numbers in  DataSets
from sklearn.preprocessing import LabelEncoder
#ٍSteps of algorithm
"""

#Create object from class LabelEncoder
LE=LabelEncoder()
#Fit to specific column
LE.fit(dataset['NameOfColumn'])
#Transform it
LE.transform(dataset['NameOfColumn'])
#update Column
dataset['NewColumn']=LE.transform(dataset['NameOfColumn'])

"""
#Example

from sklearn.preprocessing import LabelEncoder
import pandas as pd

raw_data = {'patient': [1, 1, 1, 2, 2],
            'obs': [1, 2, 3, 1, 2],
            'treatment': [0, 1, 0, 1, 0],
            'score': ['strong', 'weak', 'normal', 'weak', 'strong']}
df = pd.DataFrame(raw_data, columns=['patient', 'obs', 'treatment', 'score'])

print('Original dataframe is : \n', df)

# Create a label (category) encoder object
le = LabelEncoder()
# Fit the encoder to the pandas column
le.fit(df['score'])
#Print what is text that sound to be replaced
print('classed found : ', list(le.classes_))

# column data after replace text with numbers
print('equivilant numbers are : ', le.transform(df['score']))
#add new column with numbers instead of text
df['newscore'] = le.transform(df['score'])

#Drop old column Score
df.drop('score',axis='columns',inplace=True)

#Print dataFrame
print('Updates dataframe is : \n', df)

#-----------
#another Example
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv(r'D:\ML\SKLearn Library\Slides && Data\Data\2.16 NLP\mall.csv')
#print(data.head())
df = pd.DataFrame(data)
print('Original dataframe is : \n' ,df )
enc  = LabelEncoder()
enc.fit(df['Genre'])
print('classed found : ' , list(enc.classes_))

print('equivilant numbers are : ' ,enc.transform(df['Genre']) )

df['Genre Code'] = enc.transform(df['Genre'])
print('Updates dataframe is : \n' ,df )

print('Inverse Transform  : ' ,list(enc.inverse_transform([1,0,1,1,0,0])))

#------------------------------------------------------------------
#ugualy is faster than LabelEncoder
#2-OneHotEncoder : Create Column for every type in text column and put 1 for select type and other with 0
#لصناعة مصفوفة الواحد من النصوص
#و هي تقوم بتحويل العمود الذي يحتوي علي نصوص الي عدد من الأعمدة الجديدة , يساوي عدد الكلمات المختلفة , بحيث كل عمود يكون فيه اصفار و قيمة 1 فقط عندما تتواجد القيمة

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

data = pd.read_csv(r'D:\ML\SKLearn Library\Slides && Data\Data\2.16 NLP\mall.csv')

df = pd.DataFrame(data)


print('Original dataframe is : \n' ,df )

ohe  = OneHotEncoder()
col = np.array(df['Genre'])
col = col.reshape(len(col), 1)

ohe.fit(col)

newmatrix = ohe.transform(col).toarray()
newmatrix = newmatrix.T

df['Female'] = newmatrix[0]
df['male'] = newmatrix[1]

print('Updates dataframe is : \n' ,df )
#-------------------------------------------
#3-CountVectorizer
#و هي تقوم بقراءة النصوص الأطول , و حذف الكلمات المألوفة , ثم عمل وظيفة مشابهة لوظيفة LabelEncoder , و بعدها يمكن استخدام اي خورازم معين لعمل التصنيف او التوقع

#Example to make classification if messages is positive ot negative one
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

Count_vectorizer=CountVectorizer()
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']
Count_vectorizer.fit(simple_train)


