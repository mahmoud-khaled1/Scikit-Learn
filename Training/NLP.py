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

#---------------------------------------------------
#2-OneHotEncoder


