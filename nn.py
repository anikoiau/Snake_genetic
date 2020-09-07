# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:05:20 2020

@author: soumitra
"""

import time
import numpy as np


np.random.seed()
'''
import math
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sklearn.model_selection as s


X_train = pd.read_csv("mnist_train.csv")
aux= X_train['label']
y_train=aux.to_numpy()
y_train = y_train[:, np.newaxis]
X_train = X_train.to_numpy()
X_train=X_train[:, 1:]
X_train=X_train.astype(float)
X_train = X_train.T
y_train = y_train.T
X_train = X_train/255
Y_train = np.zeros((10, y_train.shape[1]))
for i in range(60000):
    index = y_train[0][i]
    Y_train[index][i] = 1    
    
    
X_test=pd.read_csv("mnist_test.csv")
aux= X_test['label']
y_test=aux.to_numpy()
X_test = X_test.to_numpy()
X_test=X_test[:, 1:]
X_test=X_test.astype(float)
X_test = X_test.T
y_test = y_test.T
X_test = X_test/255
'''

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    
    return A, Z



def sigmoid_backwards(dA, cache):
    Z = cache
    
    s = 1/(1 + np.exp(-Z))
   # print("s shape : ", s.shape, " dA shape : ", dA.shape)

    dZ = dA * s * (1 - s)
    
    return dZ




def linear_forward(W, A, b):
    
    if b.ndim == 1:
        b = b.reshape((b.shape[0], 1))
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activate_forward(A, W, b):
    Z, linear_cache = linear_forward(W, A, b)
    A, activation_cache = sigmoid(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache




def forward_propagation(X, parameters):
    m = X.shape[1]
    caches = []
    
    L = len(parameters) // 2
    
    A_prev = X
    
    for l in range(1, L):
       #print(l)
        
       A, cache = linear_activate_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        
       caches.append(cache)
       A_prev = A
        
    AL, cache = linear_activate_forward(A_prev, parameters['W' + str(L)], parameters['b' + str(L)])
        
    caches.append(cache)

    return AL, caches
    
    # return AL



def linear_backwards(dZ, cache):
    
    A_prev, W, b = cache
    m = dZ.shape[1]

    dA_prev = (W.T @ dZ)/m
    dW = (dZ @ A_prev.T)/m
    db = (np.sum(dZ, axis = 1, keepdims = True))/m
    
    return dA_prev, dW, db



def linear_activation_backward(dA, cache):
    
    linear_cache, activation_cache = cache
    
   # print("shape of dz : ", activation_cache.shape)
    
    dZ = sigmoid_backwards(dA, activation_cache)
    
    dA_prev, dW, db = linear_backwards(dZ, linear_cache)
    
    
    return dA_prev, dW, db




def backward_propagation(AL, y, caches):
    grads = {}
    L = len(caches)
    
  #
        
    dAL = -(y/AL - (1 - y)/(1 - AL))
    
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, caches[L - 1])
    
    for l in reversed(range(L - 1)):
       # print(l, " cache[l] shape = ", caches[l].shape)
        grads['dA' + str(l)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = linear_activation_backward(grads['dA' + str(l + 1)], caches[l])
        
        
        
    return grads



def initialize(layers):
    L = len(layers)
    
    parameters = {}
    
    # np.random.seed(int(time.time()))
    
    for l in range(1, L):
        # parameters['W' + str(l)] = np.random.rand(layers[l], layers[l - 1]) * np.sqrt(2/(layers[l - 1] + layers[l]))
        # parameters['b' + str(l)] = np.zeros((layers[l], 1))
        
        parameters['W' + str(l)] = np.random.rand(layers[l], layers[l - 1]) * 2 - 1
        parameters['b' + str(l)] = np.random.rand(layers[l], 1) * 2 - 1
        
    return parameters





def cost(AL, y):
    m = y.shape[1]
    #AL = np.argmax(AL,axis=0)+1
    J = (-1/m) * np.sum(y * np.log(AL) + (1 - y) * np.log(1 - AL))
    
    return J




def update_parametrs(grads, parameters, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]


    return parameters


def predict(X, parameters):
    L = len(parameters) // 2
    
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis = 0)
    
    return predictions


def random_minibatch(X, y, batch_size, seed):
    np.random.seed(seed)
    m = X.shape[1]
    
    indices = list(np.random.permutation(m))
    
    X_shuffled = X[:, indices]
    y_shuffled = y[:, indices]
    
    mini_batches = []
    
    mini_batch_num = m // batch_size
    
    for i in range(mini_batch_num):
        mini_batch_x = X_shuffled[:, i * batch_size:(i + 1) * batch_size]
        mini_batch_y = y_shuffled[:, i * batch_size:(i + 1) * batch_size]
        
        mini_batch = (mini_batch_x, mini_batch_y)
        
        mini_batches.append(mini_batch)
        
        
    if m % batch_size != 0:
        mini_batch_x = X_shuffled[:, mini_batch_num * batch_size:]
        mini_batch_y = y_shuffled[:, mini_batch_num * batch_size:]
        
        mini_batch = (mini_batch_x, mini_batch_y)
        
        mini_batches.append(mini_batch)
        
        
    return mini_batches
    
    
'''    

X = np.genfromtxt('1.csv', delimiter = ',')

Y = X[:, -1]

X_train = X[:, :-1].T

Y_train = np.zeros((X.shape[0], 3))

for i in [0, 1, 2]:
    Y_train[:, i] = np.where(Y == i, 1, 0)
    
Y_train = Y_train.T



costs = []
parameters = initialize([5, 15, 3])


ite = 400
learning_rate = .1
seed = ite + 10



for i in range(ite):
    seed = seed - 1
    c = 0
    
    mini_batches = random_minibatch(X_train, Y_train, 32, seed)
    
    for minibatch in mini_batches:
        
        minibatch_x, minibatch_y = minibatch
        AL, caches = forward_propagation(minibatch_x, parameters)
        #print(len(caches))
        #print("caches shape : ", caches.shape, "\ncaches : ", caches)
        #print(caches)
        
        c = c + cost(AL, minibatch_y)
        
        
        
        grads = backward_propagation(AL, minibatch_y, caches)
        
        
        
        parameters = update_parametrs(grads, parameters, learning_rate)


    
    preds_per_epoch = predict(X_train, parameters)
    print(np.sum(preds_per_epoch[np.newaxis, :] == Y_train)*100/Y_train.shape[1])
    costs.append(c)
    
    #c = cost(AL, minibatch_y)
    #print(f"epoch {i} done")
    print ("Cost after epoch %i: %f" %(i, c/Y_train.shape[1]))
    #if i % 100 == 0:
       # costs.append(c)


preds_on_train = predict(X_train, parameters)
# preds_on_test = predict(X_test, parameters)

W1 = parameters['W1']
W2 = parameters['W2']
b1 = parameters['b1']
b2 = parameters['b2']

np.savetxt('w1.csv', W1, delimiter=',')
np.savetxt('w2.csv', W2, delimiter=',')
np.savetxt('b1.csv', b1, delimiter=',')
np.savetxt('b2.csv', b2, delimiter=',')

print(np.sum(preds_on_train[np.newaxis, :] == Y_train)*100/Y_train.shape[1])
# print(np.sum(preds_on_test[np.newaxis, :] == y_test)*100/10000)


plt.plot(costs)
plt.show()

'''
   


