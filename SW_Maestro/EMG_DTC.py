import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.svm import SVC
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import projections
from math import gamma
from numpy import float64

X_data = []
y_data = []
names = ['emg1','emg2','emg3']
myList = []
df = pd.DataFrame()
frame = []

for gesture in range(1,6): 
    file = 'gesture%d.csv' %gesture
    frame = read_csv(file, names = names, skiprows=1, sep=',')
    frame['gesture'] = gesture
    myList.append(frame)

df = pd.concat(myList)
print(df)

X_data = df.drop('gesture', axis=1)
X_data = X_data.astype(float64)
X_data = X_data.fillna(X_data.median(axis=0))

y_data = df['gesture'] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 42)

# Shape of learning data set
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes = (20,20,20), learning_rate_init = 0.0001, max_iter = 200000, momentum = 0.5)

from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 25, random_state = 42)

# clf = SVC(C=1.0000, gamma=0.10000, max_iter=100)

clf.fit(X_train_std, y_train)
print("Training Set Accuracy: {:.3f}".format(clf.score(X_train_std, y_train)))
print("Test Set Accuracy: {:.3f}".format(clf.score(X_test_std, y_test)))

predictions = clf.predict(X_test_std)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions)) 
print("\nAcurracy")
print(accuracy_score(y_test, predictions))

df_ = pd.DataFrame()
df_['y_test'] = y_test
df_['predictions'] = predictions
print(df_)
