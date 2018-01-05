import numpy as np

class KNNClassifier(object):
    def __init__(self):
        self.X_Traing = None
        self.Y_Train = None

    def train(self,X_Train,Y_Train):
        self.X_Traing = X_Train
        self.Y_Train = Y_Train
    def GetDisTance(self,l,X_Test):

        dis = np.zeros((X_Test.shape[0],self.X_Traing.shape[0]))
        if l == 1:
            for i in range(X_Test.shape[0]):
                dis[i,:] = np.linalg.norm(self.X_Traing - X_Test[i,:],axis = 1)
            return dis
        if l == 0:
            M = np.dot(X_Test,self.X_Traing.T)
            te = np.square(X_Test).sum(axis = 1)
            tr = np.square(self.X_Traing).sum(axis = 1)
            dis = np.array(np.sqrt(-2 * M + tr + np.matrix(te).T))
        return dis


    def GetLabels(self,k,X_Test):
        return None