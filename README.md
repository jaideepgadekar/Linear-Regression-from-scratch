# Linear-Regression-from-scratch-with-python
In this project I used linear regression to predict yearly amount spent by customers on e-fashion startup. All necessary functions are written from scratch in python using numpy library

- **train_X_lr.csv and train_Y_lr.csv** - data for training
- Attributes for trianing(**train_X**) = Time spent on website, Duration of membership, Time spent on App, Session Duration.
- **train_Y** = Money spent on website.

## train.py

In this file, I have trained the model on given data.
```
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pickle
```
Loaded necessary libraries. PolynomialFeatures can be used to train polynomial-regression on the data to get more accuracy.

```
#loading data
def import_data():
    train_X = np.genfromtxt('train_X_lr.csv', delimiter = ',',dtype = np.float128, skip_header=1)
    train_Y = np.genfromtxt('train_Y_lr.csv',delimiter = ',', dtype = np.float128)
    return train_X,train_Y
```

```

def compute_cost(X,Y,W):
    pred_Y = np.dot(X,W)
    mse = np.sum(np.square(pred_Y - Y))
    cost_value = mse/(2*len(X))
    return cost_value
```
```
#Gradients of  cost function
def compute_gradient_cost_function(X,Y,W):
    pred_Y = np.dot(X,W)
    diff = pred_Y - Y
    dW = (1/len(X))*(np.dot(diff.T, X))
    dW = dW.T
    return dW
```
```
#updating parameters
def optimist_weights_grad(X,Y,W,num_iter,learning_rate):
    for i in range(num_iter):
        dW = compute_gradient_cost_function(X, Y, W) 
        W -= learning_rate*dW
        cost = compute_cost(X,Y,W)
        print(i,cost)
    return W
```
```
def train_model(X,Y):
#training the model with desired polynomial regression
    X = np.insert(X,0,1, axis = 1)
    poly = PolynomialFeatures(6)
    X = poly.fit_transform(X)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    W = optimist_weights_grad(X, Y, W, 20000, 1e-23)
    return W
```    
        
```

if __name__ == '__main__':
    X, Y = import_data()
    W = train_model(X, Y)
#saving the weights caclulated
    with open('weights.pkl' , 'wb') as f:
        pickle.dump(W, f)
```  

## predict.py

In this file, I have wrote prediction  function.

```
import numpy as np
import csv
import sys
import pickle
from sklearn.preprocessing import PolynomialFeatures
```
```
#importing data from paths
def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float128, skip_header=1)
    with open(weights_file_path, 'rb') as f:
        weights = pickle.load(f)
    return test_X, weights
```
```
#predicting label with trained weights
def predict_target_values(test_X, weights):
    test_X = np.insert(test_X, 0, 1, axis = 1)
    poly = PolynomialFeatures(6)
    test_X = poly.fit_transform(test_X)
    pred_Y = np.dot(test_X, weights)
    return pred_Y
```
```
#saving predicted values to csv file
def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()
```
```

def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "weights.pkl")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lr.csv")
```
```
if __name__ == "__main__":
#taking input test_X file path   
   test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
```


