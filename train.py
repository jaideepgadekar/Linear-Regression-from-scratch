#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:38:42 2020

@author: jaideep
"""


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pickle

def import_data():
    train_X = np.genfromtxt('train_X_lr.csv', delimiter = ',',dtype = np.float128, skip_header=1)
    train_Y = np.genfromtxt('train_Y_lr.csv',delimiter = ',', dtype = np.float128)
    return train_X,train_Y

def compute_cost(X,Y,W):
    pred_Y = np.dot(X,W)
    mse = np.sum(np.square(pred_Y - Y))
    cost_value = mse/(2*len(X))
    return cost_value

def compute_gradient_cost_function(X,Y,W):
    pred_Y = np.dot(X,W)
    diff = pred_Y - Y
    dW = (1/len(X))*(np.dot(diff.T, X))
    dW = dW.T
    return dW

def optimist_weights_grad(X,Y,W,num_iter,learning_rate):
    for i in range(num_iter):
        dW = compute_gradient_cost_function(X, Y, W) 
        W -= learning_rate*dW
        cost = compute_cost(X,Y,W)
        print(i,cost)
    return W
    
def train_model(X,Y):
    X = np.insert(X,0,1, axis = 1)
    poly = PolynomialFeatures(6)
    X = poly.fit_transform(X)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    W = optimist_weights_grad(X, Y, W, 20000, 1e-23)
    return W
    
        

if __name__ == '__main__':
    X, Y = import_data()
    W = train_model(X, Y)
    with open('weights.pkl' , 'wb') as f:
        pickle.dump(W, f)