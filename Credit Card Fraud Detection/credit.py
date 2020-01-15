import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\anush\Downloads\creditcard.csv\creditcard.csv")
#
 corr = df.corr()
# # ax = sns.heatmap(corr)
# # ax.set_xticklabels(
# #     ax.get_xticklabels(),
# #     rotation=45,
# #     horizontalalignment='right'
# # )
# mask = np.zeros_like(df, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# sns.set_style(style = 'white')
#
#     # Set up  matplotlib figure
# f, ax = plt.subplots(figsize(20,20))
#
#     # Add diverging colormap from red to blue
# # cmap = sns.diverging_palette(250, 10, as_cmap=True)
# # cmap = sns.diverging_palette(,as_cmap=True)
# sns.heatmap(df, mask=mask,
#                 square=True,
#                 linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
#
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#
# df1 = df.drop(['V13','V24','V27','V28','V15','V26'], axis=1)
# corr = df1.corr()
#
# hist = df1['Class']
# plt.hist(hist)

X = df.iloc[:,:30].values
y = df.iloc[:,30].values

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

df2 = pd.read_csv(r"C:\Users\anush\Downloads\zeros.csv")
X = df2.iloc[:,:30].values
y = df2.iloc[:,30].values

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

plt.scatter(df.Amount, df.Time)
