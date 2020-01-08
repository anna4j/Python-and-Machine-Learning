import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from copy import deepcopy
import seaborn as sns; sns.set()

df = pd.read_csv(r"C:\Users\anush\Downloads\Social_Network_Ads.csv")

X = df.drop(['User ID', 'Gender', 'Purchased'], axis = 1)
y = df['Purchased']

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
y_pred = Kmean.predict(X)
print(Kmean.labels_)
df2 = pd.DataFrame(y_pred)
df2.head()
print(confusion_matrix(y, y_pred))
print(accuracy_score(y, y_pred))

df_col = pd.concat([X,df2], axis=1)
df_col.head()

ax = sns.scatterplot(x="Age", y="EstimatedSalary", data=df_col)
ax = sns.scatterplot(x="Age", y="EstimatedSalary", data=df)