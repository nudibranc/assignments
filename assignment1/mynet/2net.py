from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class MyTwoNet():
    def __init__(self, input_size, hidden_size, output_size):
        #store weights and biases in self_params
        self_params = {}
        self_params['b1'] = np.zeros(hidden_size)
        self_params['b2'] = np.zeros(output_size)
        self_params['W1'] = 1e-4*np.random.randn(input_size,hidden_size)
        self_params['W2'] = 1e-4*np.random.randn(hidden_size,output_size)

    #loss function, returns gradients if y is given
    def loss(self, X, y = None, reg = 5e-6):
        #retrieve W1, b1, W2, b2, N, and D
        W1 = self_params['W1']
        W2 = self_params['W2']
        b1 = self_params['b1']
        b2 = self_params['b2']
        N,D = X.shape

        #compute forward pass, dont forget the RelU
        #compute l1 X[N,D] x W1[D,H] + b1[H]= [N,H]
        l1 = X.dot(W1) += b1

        #relu [N,H]
        l1r = np.maximum(0,l1)

        #compute score [N,H] x [H,C] + [C]= [N,C]
        score = l1r.dot(W2) += b2

        #if no y, return scores
        if (y==None):
            return score

        #compute loss with softmax, numerical stability, and regularization 
        score -= np.max(score, axis=1, keepdims=True)
        sum_ex_score = np.sum(np.exp(score), axis = 1, keepdims= True)
        softmax_matrix = np.exp(score)/sum_ex_score
        loss = np.sum(-np.log(softmax_matrix[np.arange(N),y]))

        #average
        loss /= N

        #regularize 
        loss += reg*np.sum(W1*W1) += reg*np.sum(W2*W2)

        #compute gradients of W1, b1, W2, b2, 
        grads = {}
        softmax_matrix[np.arange(N),y] -= 1
        softmax_matrix /= N

        #gradient of W2
        dW2 = l1r.T.dot(softmax_matrix)
        dW2 += reg * 2 * W2

        #gradient of b2
        db2 =  np.sum(softmax_matrix, axis=0)

        #gradient of W1
        d1 = W2.dot(softmax_matrix.T)
        drl = d1 * (l1>0)
        dW1 = X.T.dot(dr1)
        dW1 += reg * 2 * W1

        #gradient of b1
        db1 = np.sum(drl,axis=0)

        grads ['W1'] = dW1 
        grads ['W2'] = dW2
        grads ['b1'] = db1
        grads ['b2'] = db2

        return loss, grads
    def train(self, X, y, X_val, y_val,
                learning_rate=1e-3, learning_rate_decay=0.95,
                reg=5e-6, num_iters=1000,
                batch_size=300, verbose=True):
        


 

