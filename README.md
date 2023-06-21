# Iris Flower Prediction App
### Web App that predicts the class of Iris Flower based on measurements of the petals and sepals using Decision Trees 
''
''
- The model used is Decision Tree Classifier
- The dataset is taken from 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

## 1. Splitting the Dataset & Training the Model
> X = irisData.drop('Class', axis=1)
>  
> Y = irisData['Class']
>  
> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
>  
> DTC = DecisionTreeClassifier(max_leaf_nodes=3, random_state=42)
>  
> DTC.fit(X_train, Y_train)
>  
