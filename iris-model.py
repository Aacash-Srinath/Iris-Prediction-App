import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import joblib


csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
irisData =  pd.read_csv(csv_url, names=col)


X = irisData.drop('Class', axis=1)
Y = irisData['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=42)
model.fit(X_train, Y_train)

model.predict(X_test)

joblib.dump(model, 'Iris-Prediction/DTC.pkl')