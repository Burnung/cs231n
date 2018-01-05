import numpy as np
from cs231n.data_utils import load_CIFAR10
import random
import matplotlib.pyplot as plt
from cs231n.MyClassifier import KNNClassifier
from cs231n.MyClassifier import SoftMax
from cs231n.MyClassifier import MSVM
import time

plt.rcParams['figure.figsize'] = (10.0,8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

def Getcifar10Data(TrainNum = 49000,ValidationNum = 1000,TestNum = 1000):
    x_train,y_train,x_test,y_test = load_CIFAR10(cifar10_dir)
    mask = list(range(TrainNum,ValidationNum +TrainNum))
    x_Validation = x_train[mask]
    y_Validation = y_train[mask]

    mask = list(range(TrainNum))
    x_Train = x_train[mask]
    y_Train = y_train[mask]

    mask = list(range(TestNum))
    x_Test = x_test[mask]
    y_Test = y_test[mask]

    x_Train = np.reshape(x_Train,(x_Train.shape[0],-1))
    y_Train = np.reshape(y_Train,(y_Train.shape[0],-1))

    x_Validation = np.reshape(x_Validation,(x_Validation.shape[0],-1))
    y_Validation = np.reshape(y_Validation,(y_Validation.shape[0],-1))

    x_Test = np.reshape(x_Test,(x_Test.shape[0],-1))
    y_Test = np.reshape(y_Test,(y_Test.shape[0],-1))

    print('x_Train shape is ',x_Train.shape)
    print('y_train shape is ',y_Train.shape)

    x_Train = np.hstack([x_Train,np.ones((x_Train.shape[0],1))])
    x_Test = np.hstack([x_Test,np.ones((x_Test.shape[0],1))])
    x_Validation = np.hstack([x_Validation,np.ones((x_Validation.shape[0],1))])

    print('x_Train shape is ',x_Train.shape)
    print('y_train shape is ',y_Train.shape,y_Train.dtype)
    print('Get cifar10 Data ok')

    return x_Train,y_Train,x_Validation,y_Validation,x_Test,y_Test

x_train,y_train,x_Validation,y_Validation,x_test,y_test = Getcifar10Data()#5000,1000,10)

#mySVM = MSVM(x_train,y_train,lamd = 0.01,numCatagory = 10)
#mySVM.DoTrain(eta = 0.0000001, X_Validation = x_test,Y_Validation = y_test)
mySoftmax = SoftMax(x_train,y_train,lamd = 0.001)
mySoftmax.Optiminal(eta = 0.0000001,echo = 30,batchSize = 64,X_Validation = x_test,Y_Validation = y_test)