import numpy as np
from random import shuffle
import math

class SoftMax(object):
  def __init__(self,x_train,y_train,lamd = 0.00001):
    self.x_train = x_train
    self.y_train = y_train
    self.w = np.random.randn(x_train.shape[1],10) * 0.0001
    self.lamd = lamd

  def getLossAndGrad(self,X,Y):
    retY = np.dot(X,self.w)
    rowMax = np.max(retY,axis = 1).reshape(retY.shape[0],1)
    probY = np.exp(retY - rowMax) / np.sum(np.exp(retY - rowMax),axis = 1,keepdims = True)
    maskMat = np.zeros_like(probY)
    Y = Y.astype(int)
    maskMat[range(retY.shape[0]),Y] = 1.0
    sum_Y = probY[list(range(retY.shape[0])),Y]
    loss = np.sum(-np.log(sum_Y)) / retY.shape[0] + self.lamd * np.sum(self.w *self.w)
    dw = np.zeros_like(self.w)
    dw = -np.dot(X.T,maskMat - probY ) / retY.shape[0] + self.lamd * self.w
    return loss, dw

  def Optiminal(self,eta = 0.001,echo = 30,batchSize = 64,X_Validation = None,Y_Validation = None):
    Train_data = np.hstack([self.x_train,self.y_train])
    dataSize = self.y_train.shape[0]
    batchNum = math.floor(dataSize / batchSize)
    print('begin optinal...')
    for i in range(echo):
      np.random.shuffle(Train_data)
      loss = 0.0
      for nBatch in range(batchNum):
        mask = list(range(nBatch * batchSize, nBatch * batchSize + batchSize))
        x_train = Train_data[mask,:-1]
        y_train = Train_data[mask,-1]
        tLoss,dw = self.getLossAndGrad(x_train,y_train)
        loss += tLoss
        self.w -= (eta * dw)
      mask = list(range(batchNum * batchSize,dataSize))
      x_train = Train_data[mask,:-1]
      y_train = Train_data[mask,-1]
      tLoss,dw = self.getLossAndGrad(x_train,y_train)
      loss += tLoss
      self.w -= eta * dw
      loss /= (batchNum + 1)
      print('ehoc ',i,'loss is: ',loss)
    if not X_Validation is None and not Y_Validation is None :
      p_y = self.predict(X_Validation)
      all_e = 0
      for i in range(p_y.shape[0]):
        if p_y[i] == Y_Validation[i]:
          all_e += 1
      print('accuracy rate is  ',all_e / p_y.shape[0])

  def predict(self,X):
    print('predict X shape is',X.shape)
    p_r = np.argmax(np.dot(X,self.w),axis = 1)
    print('prediction size is ',p_r.shape)
    return p_r





