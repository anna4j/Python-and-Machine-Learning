import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import os

df = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\Python-and-Machine-Learning\wine_quality.csv")
#df.head()
#df.shape
#df.isnull().any()
df['grade'] = 1 # good
df.grade[df.quality < 7] = 0 # not good0

plt.figure(figsize = (8,8))
labels = df['grade'].value_counts().index
plt.pie(df['grade'].value_counts(), autopct='%1.1f%%')
plt.legend(labels, loc="best")
plt.axis('equal')
plt.title('Quality Pie Chart')
#plt.show()
#print('The good quality wines count for ',round(df.grade.value_counts(normalize=True)[1]*100,1),'%.')

#correlation heatmap
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#plt.subplots(figsize = (12,12))
sns.heatmap(df.corr(), annot=True, mask = mask, cmap = 'RdBu_r', linewidths=0.1, linecolor='white', vmax = .9,square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
#plt.show()

#sns.scatterplot(x= 'quality', y='fixed acidity', data=df)
#df.loc[df['fixed acidity']>14,'fixed acidity']=np.median(df['fixed acidity'])

#sns.scatterplot(x= 'quality', y='citric acid', data=df)
#df.loc[df['citric acid']>1.50,'citric acid']=np.median(df['citric acid'])

#sns.scatterplot(x= 'quality', y='residual sugar', data=df)
#df.loc[df['residual sugar']>60,'residual sugar']=np.median(df['residual sugar'])

#sns.scatterplot(x= 'quality', y='residual sugar', data=df)
#df.loc[df['residual sugar']>60,'residual sugar']=np.median(df['residual sugar'])

#sns.scatterplot(x= 'quality', y='free sulfur dioxide', data=df)
#df.loc[df['free sulfur dioxide']>250,'free sulfur dioxide']=np.median(df['free sulfur dioxide'])

#sns.scatterplot(x= 'quality', y='density', data=df)
#df.loc[df['density']>1.03,'density']=np.median(df['density'])

#corr = df.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

X = df.drop(['quality','grade'], axis = 1)
n=11
y = df['grade']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=7)

x_train_check = X.values.reshape((len(X), n))
x_test_check = df['grade'].values.reshape((len(df['grade']), 1))

x_train_mat = X_train.values.reshape((len(X_train), n))
#x_test_mat = X_test.values.reshape((len(X_test), n))


classifier = RandomForestClassifier(n_estimators = 50, random_state = 20)
classifier.fit(x_train_mat,y_train)
#y_pred = classifier.predict(x_test_mat)

yy_pred = classifier.predict(x_train_check)
print(accuracy_score(x_test_check, yy_pred)*100,'%')
#print(confusion_matrix(x_test_check, yy_pred))
#print(accuracy_score(y_test,y_pred))
#print(classification_report(y_test, y_pred))

#k = [100,200,300,325, 350,400,500]
#for i in k:
 #   rf_tune = RandomForestClassifier(n_estimators=i, random_state=50)
  #  rf_tune.fit(x_train_mat,y_train)
   # y_pred = rf_tune.predict(x_test_mat)
    #print(accuracy_score(y_test, y_pred)*100,'%')

#x_train_check = X.values.reshape((len(X), n))
#x_test_check = df['grade'].values.reshape((len(df['grade']), 1))

#k = [10,20,30,40,50]
f#or i in k:
  #  rf_tune = RandomForestClassifier(n_estimators=50, random_state=i)
   # rf_tune.fit(x_train_mat,y_train)
    #yy_pred = rf_tune.predict(x_train_check)
    #print(accuracy_score(x_test_check, yy_pred)*100,'%')


