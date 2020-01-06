import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv(r"C:\Users\anush\Downloads\Social_Network_Ads.csv")
#df = pd.get_dummies(df)
#corr = df.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#df.head()
df1 = df.drop(['User ID', 'Gender'], axis = 1)

#corr = df1.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#df.describe()
#sns.set_style("whitegrid")
#sns.boxplot(x='Purchased', y='Gender_Female', data=df1)
#df1['Purchased'].value_counts()
df.head()
df = pd.get_dummies(df)
X = df.iloc[:,:5]
y = df.iloc[:,5]
#X = df.iloc[:,:4].values
# y = df.iloc[:,4].values
classifier = RandomForestClassifier(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score :')
score = accuracy_score(y_test, y_pred)
print (score)



