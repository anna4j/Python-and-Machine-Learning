import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv(r"C:\Users\anush\Downloads\Social_Network_Ads.csv")
#df = df.drop(['Gender'], axis = 1)
df.loc[df['Age']>55,'Age']=np.median(df['Age'])
df.loc[df['EstimatedSalary']>120000,'EstimatedSalary']=np.mean(df['EstimatedSalary'])

#enc = LabelEncoder()
#df['Gender'] = enc.fit_transform(df['Gender'])
X = df.drop(['User ID', 'Gender'], axis = 1)
#y = df['Purchased']
X = df.iloc[:,2:3].values
y = df.iloc[:,4].values
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)


classifier = RandomForestClassifier(n_estimators = 1000, max_features = 'auto', max_depth = 140, criterion = 'entropy')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))

scores = []
#best_svr = RandomForestClassifier(n_estimators = 1000, max_features = 'auto', max_depth = 140, criterion = 'entropy')
cv = KFold(n_splits=10, shuffle=True, random_state = 42)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    classifier.fit(X_train, y_train)
    scores.append(classifier.score(X_test, y_test))
print(cross_val_score(classifier, X, y, cv=10, scoring='roc_auc'))

#from sklearn.metrics import roc_curve, auc
#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#roc_auc = auc(false_positive_rate, true_positive_rate)
#roc_auc





# number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
#max_features = ['auto', 'sqrt']

# max depth
#max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
#max_depth.append(None)
# create random grid
#random_grid = { 'n_estimators': n_estimators,  'max_features': max_features,'max_depth': max_depth }
# Random search of parameters
#rfc_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
#rfc_random.fit(X_train, y_train)
# print results
#print(rfc_random.best_params_)

