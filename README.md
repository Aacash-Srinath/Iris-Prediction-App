# Iris Flower Prediction App (Model OVerview)
### Web App that predicts the class of Iris Flower based on measurements of the petals and sepals using Decision Trees 
 
 
- The model used is Decision Tree Classifier
- The dataset is taken from 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

## 1. Importing the Dataset & Naming the Columns
> csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
>  
> col = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
>  
> irisData =  pd.read_csv(csv_url, names = col)

## 2. Splitting the Dataset & Training the Model
> X = irisData.drop('Class', axis=1)
>  
> Y = irisData['Class']
>  
> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
>  
> DTC = DecisionTreeClassifier(max_leaf_nodes=3, random_state=42)
>  
> DTC.fit(X_train, Y_train)

## 3. Decision Tree of the Given Dataset
![download](https://github.com/Aacash-Srinath/Iris-Prediction-App/assets/100955640/7e629f7b-de39-4a72-9af0-ca38ebab59b7)

## 4. Testing the Accuracy of the Model
> Y_pred = DTC.predict(X_test)
> 
> accuracy = accuracy_score(Y_test, Y_pred) * 100
>
> print(f"Accuracy : {accuracy}")
>  > ### The Model has an Accuracy of 97.37%











