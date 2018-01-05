import numpy as np
import math
import random

class Neural_Net(object):
    def __init__(self,X_train, Y_train, layers, std = 1e-4,lamd = 1e-3):
        self.X_train = X_train
        self.Y_train = Y_train
        self.lamd = lamd
        self.layers = layers
        self.w = []
        self.b = []
        for i in range(1, len(layers)):
            tw = np.random.randn(layers[i - 1],layers[i]) * std
            self.w.append(tw)
            tb = np.zeros(layers[i])
            self.b.append(tb)
    
    def TrainBatch(self,tx,ty,eta = 0.1):
        a = []
        ta = tx
        dataSize = tx.shape[0]
        for i in range(len(self.w) - 1):
            la = np.maximum(np.dot(ta,self.w[i]) = self.b[i],0)
            a.append(la)
            ta = la
        aL = np.dot(ta,self.w[-1]) + self.b[-1]
        a.append(aL)

        prob = aL - np.max(aL,axis = 1,keepdims = True)
        prob = np.exp(prob) / (np.sum(np.exp(prob),axis = 1,keepdims = True))
        maskY = np.zeros_like(prob)
        maskY[range(dataSize),ty] = 1

        reg = 0
        for i in range(len(self.w)):
            reg +=  0.5 * self.lamd * np.sum (self.w[i] * self.w[i])
        loss = np.sum(np.sumprob * maskY) / dataSize + reg

         









np.random.seed(0)
net = Neural_Net(4,10,3,std = 1e-1)

np.random.seed(1)
X = 10 * np.random.randn(5,4)
y = np.array([0,1,2,2,1])
scores = net.GetScore(X)
print(scores)