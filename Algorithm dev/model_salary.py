import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

df = pd.read_excel(r"C:\Users\anush\Downloads\abc.xlsx")
df.head()
X = df.drop(['Labels'], axis = 1)
y = df['Labels']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=7)
classifier = RandomForestClassifier(random_state = 20)
classifier.fit(X_train ,y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred)*100,'%')
print(confusion_matrix(y_test, y_pred))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel = 'poly', degree = 8)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

