import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv(r"C:\Users\anush\Downloads\Social_Network_Ads.csv")
X = df.iloc[:,1:4].values
y = df.iloc[:,4].values

x = df.iloc[:,[1]].values
enc = LabelEncoder()
X[:,0] = enc.fit_transform(x[:,0])

classifier = RandomForestClassifier(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)
results = confusion_matrix(y_test, y_pred)

cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred)


#SVM CLASSIFIER
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

#KERNEL SVM POLYNOMIAL
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

#GAUSSIAN
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

#SIGMOID
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

#LOGISTIC REGRESSION

from sklearn import linear_model
reg = linear_model.LogisticRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

df.loc[df['Age']>55,'Age']=np.mean(df['Age'])
df.loc[df['EstimatedSalary']>120000,'EstimatedSalary']=np.mean(df['EstimatedSalary'])

sns.set_style("whitegrid")
sns.boxplot(x='Purchased', y='EstimatedSalary', data=df)
sns.set_style("whitegrid")
sns.boxplot(x='Purchased', y='Age', data=df)

df.head()

X = df.iloc[:,1:3].values
y = df.iloc[:,3].values

x = df.iloc[:,[1]].values
enc = LabelEncoder()
X[:,0] = enc.fit_transform(x[:,0])
df = df.drop(['Gender'], axis = 1)

classifier = RandomForestClassifier(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
