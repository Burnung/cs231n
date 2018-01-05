import numpy as np
import random
import math
class MSVM(object):
    def __init__(self,x_Train,y_Train,lamd,numCatagory):
        self.x_Train = x_Train
        self.y_Trian = y_Train
        self.lamd = lamd
        self.w = np.random.randn(x_Train.shape[1],numCatagory) * 0.001
    def GetLossAndGrade(self,X,Y):
        dataSize = Y.shape[0]
        tY = np.dot(X,self.w)
        rightY = tY[np.arange(dataSize),Y].reshape(dataSize,1)
        lossMat = tY - rightY + 1.0
        lossMat[np.arange(dataSize),Y] = 0.0
        lossMat[lossMat < 0.0] = 0.0
        loss = np.sum(lossMat) / dataSize + self.lamd * np.sum(self.w *self.w)


        dw = np.zeros_like(self.w)
        lossMat[lossMat > 0.0] = 1.0
        row_sum = np.sum(lossMat,axis = 1)
        lossMat[np.arange(dataSize),Y] = -row_sum
        dw += np.dot(X.T,lossMat) / dataSize + self.lamd * self.w
        return loss,dw

    def Predict(self,X):
        Y = np.dot(X,self.w)
        retY = np.argmax(Y,axis = 1)
        return retY

    def DoTrain(self,echo = 30,eta = 0.001,batchsize = 64,X_Validation = None,Y_Validation = None):
        print('Begin Training...')
        Datasize = self.x_Train.shape[0]
        trainData = np.hstack([self.x_Train,self.y_Trian])
        batchNum = math.floor(Datasize / batchsize)
        for i in range(echo):
            np.random.shuffle(trainData)
            loss = 0.0
            for nowBatch in range(batchNum):
                mask = list(range(nowBatch * batchsize,nowBatch * batchsize + batchsize))
                tx = trainData[mask,:-1]
                ty = trainData[mask,-1].astype('int32')
                tloss ,tgrade = self.GetLossAndGrade(tx,ty)
                self.w -= eta * tgrade
                loss += tloss
            mask = list(range(batchNum * batchsize,Datasize))
            tx = trainData[mask,:-1]
            ty = trainData[mask,-1].astype('int32')
            tloss,grade = self.GetLossAndGrade(tx,ty)
            self.w -= eta * grade
            loss += tloss
            print('echo %d ,error is %f'%(i + 1, loss /(batchNum + 1)))
        print('training over')
        if not X_Validation is None and not Y_Validation is None:
            py = self.Predict(X_Validation)
            ret = 0
            for i in range(py.shape[0]):
                if py[i] == Y_Validation[i]:
                    ret += 1
            print('accuracy rate is ',ret / py.shape[0])

            
            



