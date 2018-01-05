import numpy as np
import math
import matplotlib.pyplot as plt

N = 300
D = 2
K = 3


def Generatdata():
    X = np.zeros((N*K,D))
    Y = np.zeros(N*K)
    for j in range(K):
        xi = range(N*j, N*(j+1))
        r = np.linspace(0.0,1.0,N)
        t = np.linspace(j*4, (j+1)*4,N) + np.random.randn(N) * 0.2
        X[xi] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[xi] = j
    plt.scatter(X[:,0],X[:,1],c = Y,s = 40)
    return X,Y

class SofMax(object):
    def __init__(self,X_train,Y_train,k,lamd = 0.001):
        self.X_train = X_train
        self.Y_train = Y_train
        self.w = np.random.randn(X_train.shape[1],k) * 0.01
        self.b = np.zeros((1,k))
        self.lamd = lamd

    def GetScore(self,x,y):
        a = np.dot(x,self.w) + self.b
        aa = a - np.max(a,axis = 1,keepdims = True)
        prob = np.exp(aa) /(np.sum(np.exp(aa),axis = 1,keepdims = True))
        scores = -np.log(prob)
        ymask = np.zeros_like(scores)
        ymask[range(x.shape[0]),y] = 1.0
        loss = np.sum(scores * ymask) / x.shape[0] + self.lamd * np.sum(self.w * self.w) * 0.5
        dw = -np.dot(x.T,ymask - prob) / x.shape[0] + self.lamd * self.w
        db = -np.sum(ymask - prob,axis = 0,keepdims = True) /x.shape[0]
        return loss,dw,db 

    def Predict(self,tX):
        retY = np.dot(tX,self.w) - self.b
        return np.argmax(retY,axis = 1)

    def Traing(self,echo = 30,batchSize = 10,eta = 0.1):
        allDataSize = self.X_train.shape[0]
        allData = np.append(self.X_train,self.Y_train.reshape(allDataSize,1),axis = 1)
        for i in range(echo):
            np.random.shuffle(allData)
            allX = allData[:,:-1]
            allY = allData[:,-1]
            bathcCount = math.floor(allDataSize / batchSize)
            loss = 0
            print('echo :',i)
            for j in range(bathcCount):
                tx = allX[j * batchSize:batchSize * (j+1),:]
                ty = allY[j*batchSize:(j+1) * batchSize].astype(int)
                #print(y.dtype)
                tloss,tdw,tdb = self.GetScore(tx,ty)
                loss += tloss
                #print(tdw)
                #print(tdb)
                self.w -= eta * tdw
                self.b -= eta * tdb
            print('loss is' ,loss / bathcCount)
        print('train over!')

class NerualNet(object):
    def __init__(self,X_train,Y_train,h,k,lamd):
        self.X_train = X_train
        self.Y_train = Y_train
        self.w1 = np.random.randn(X_train.shape[1],h) * 0.01
        self.w2 = np.random.randn(h,k) * 0.01
        self.b1 = np.zeros((1,h))
        self.b2 = np.zeros((1,k))
        self.l0 = X_train[1]
        self.l1 = h
        self.l2 = k
        self.lamd = lamd

    def trainBatch(self,tx,ty,eta = 0.01):
        dataSize = tx.shape[0]
        a1 = np.maximum(np.dot(tx,self.w1) + self.b1,0)
        a2 = np.dot(a1,self.w2) + self.b2

        a2 = a2 - np.max(a2,axis = 1,keepdims = True)
        prob = np.exp(a2) / np.sum(np.exp(a2),axis = 1,keepdims = True)
        score = -np.log(prob)

        ymask = np.zeros_like(prob)
        #print(a2.shape)
        ymask[range(dataSize),ty] = 1

        loss = np.sum(score * ymask) / dataSize + 0.5 * self.lamd *\
        (np.sum(self.w1 * self.w1 + np.sum(self.w2 * self.w2)))

        dw2 = np.dot(a1.T,prob - ymask) /dataSize + self.lamd * self.w2
        db2 = np.sum(prob - ymask,axis = 0,keepdims = True) / dataSize

        mid1 = np.dot((prob - ymask),self.w2.T)
        w1Mask = np.ones_like(mid1)
        w1Mask[a1 == 0] = 0
        mid1 = mid1 * w1Mask

        db1 = np.sum(mid1,axis = 0,keepdims = True) /dataSize
        dw1 = np.dot(tx.T,mid1) / dataSize + self.w1 *self.lamd

        self.w1 -= eta * dw1
        self.w2 -= eta * dw2
        self.b1 -= eta * db1
        self.b2 -= eta * db2 

        return loss

    def Traing(self,echo = 30,batchSize = 10,eta = 0.01):
        allDataSize = self.X_train.shape[0]
        allData = np.append(self.X_train,self.Y_train.reshape(allDataSize,1),axis = 1)
        for i in range(echo):
            np.random.shuffle(allData)
            allX = allData[:,:-1]
            allY = allData[:,-1]
            bathcCount = math.floor(allDataSize / batchSize)
            loss = 0
            print('echo :',i)
            for j in range(bathcCount):
                tx = allX[j * batchSize:batchSize * (j+1),:]
                ty = allY[j*batchSize:(j+1) * batchSize].astype(int)
                #print(y.dtype)
                tloss = self.trainBatch(tx,ty,eta)
                loss += tloss
                #print(tdw)
                #print(tdb)
            print('loss is' ,loss / bathcCount)
        print('train over!')
    
    def predict(self,tx):
        a1 = np.maximum(0,np.dot(tx,self.w1) + self.b1)
        a2 = np.dot(a1,self.w2) + self.b2
        return np.argmax(a2,axis = 1)

X,Y = Generatdata()
#plt.show()

#mySoftMax = SofMax(X,Y,K,0.005)
#mySoftMax.Traing(echo  = 200,eta = 0.1)
h = 100
myNet = NerualNet(X,Y,h,K, 0.001)
myNet.Traing( echo = 300,eta = 0.05)
predY = myNet.predict(X)
#rint(predY)
print("acurat is :",np.mean(predY == Y.astype(int)) )