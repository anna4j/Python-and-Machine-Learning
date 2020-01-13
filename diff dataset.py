import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


df = pd.read_csv(r"C:\Users\anush\Downloads\Assignment3_TrainingSet(2).csv")

#plt.hist(df['Field_info2'])
#plt.show()
#sns.boxplot('Field_info2', data=df)
#enc = LabelEncoder()
#df['QuoteConversion_Flag'].corr(df['Geographic_info2'])

df = df.drop(['Property_info2', 'Quote_ID', 'Original_Quote_Date', 'Sales_info5', 'Property_info1', 'Personal_info5','Personal_info1', 'Coverage_info2' ,'Sales_info4', 'Personal_info4' ], axis = 1)


#df['Personal_info1'] = df['Personal_info1'].astype(str)
#df['Property_info1'] = df['Property_info1'].astype(str)



enc = LabelEncoder()
df['Field_info1'] = enc.fit_transform(df['Field_info1'])
df['Field_info4'] = enc.fit_transform(df['Field_info4'])
df['Field_info3'] = enc.fit_transform(df['Field_info3'])
df['Coverage_info3'] = enc.fit_transform(df['Coverage_info3'])
#df['Sales_info4'] = enc.fit_transform(df['Sales_info4'])
#df['Personal_info1'] = enc.fit_transform(df['Personal_info1'])
df['Personal_info3'] = enc.fit_transform(df['Personal_info3'])
#df['Property_info1'] = enc.fit_transform(df['Property_info1'])
df['Property_info3'] = enc.fit_transform(df['Property_info3'])
df['Geographic_info4'] = enc.fit_transform(df['Geographic_info4'])
df['Geographic_info5'] = enc.fit_transform(df['Geographic_info5'])

#corr = df.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

#df.isnull().any()
#df['QuoteConversion_Flag'].corr(df['Property_info1'])
#df['QuoteConversion_Flag'].corr(df['Personal_info5'])
#df['QuoteConversion_Flag'].corr(df['Personal_info1'])
#df['QuoteConversion_Flag'].corr(df['Coverage_info2'])



#sns.boxplot(y='Field_info2', data=df)
#sns.boxplot(x='Coverage_info1', data=df)
#sns.boxplot(y='Coverage_info2', data=df)
#sns.boxplot(y='Sales_info2', data=df)
#sns.boxplot(y='Sales_info3', data=df)
#sns.boxplot(y='Personal_info2', data=df)
#sns.boxplot(y='Property_info5', data=df)
#sns.boxplot(y='Geographic_info1', data=df)
#sns.boxplot(y='Geographic_info3', data=df)
#df['Coverage_info2'].value_counts()

df.head()

X = df.drop(['QuoteConversion_Flag'], axis = 1)
y = df['QuoteConversion_Flag']

df['Coverage_info1'] = np.sin(df['Coverage_info1'])
df['Sales_info2'] = np.sin(df['Sales_info2'])
df['Sales_info3'] = np.sin(df['Sales_info3'])
df['Geographic_info3'] = np.sin(df['Geographic_info3'])
df['Geographic_info2'] = np.sin(df['Geographic_info2'])
df['Geographic_info1'] = np.sin(df['Geographic_info1'])
df['Property_info5'] = np.sin(df['Property_info5'])

df['Coverage_info1']

classifier = RandomForestClassifier(random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))

