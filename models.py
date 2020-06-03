import numpy as np
from collections import defaultdict
import heapq
from scipy.spatial import distance
from scipy import stats
from numpy import linalg


# TODO: You can import anything from numpy or scipy here!

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class PCA(Model):

    def __init__(self, X, target_dim):
        super().__init__()
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]
        self.target_dim = target_dim
        self.W = None

    def fit(self, X):
        Xt = X.transpose()
        for i in range(self.x_dim):
            sigma = np.sqrt(np.var(Xt[i]))
            Xt[i] = Xt[i] - np.mean(Xt[i])
            if sigma != 0:
                Xt[i] = Xt[i] / sigma
        X = Xt.transpose()
        cov_m = np.cov(X.transpose())
        # x_dim, x_dim * x_dim
        eig_val, eig_vec = np.linalg.eig(cov_m)
        eigValIndice = eig_val.argsort()[-self.target_dim:]
        tem = eig_vec[:, eigValIndice]
        data = X @ tem
        return data


class LLE(Model):

    def __init__(self, X, target_dim, lle_k):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]

        self.target_dim = target_dim
        self.k = lle_k

    def fit(self, X):
        # step 1: neighbors
        Xt = X.transpose()
        for i in range(self.x_dim):
            sigma = np.sqrt(np.var(Xt[i]))
            Xt[i] = Xt[i] - np.mean(Xt[i])
            if sigma != 0:
                Xt[i] = Xt[i] / sigma
        X = Xt.transpose()
        neighbors = []
        for i in range(len(X)):
            distance = []
            neighbors_i = []
            for j in range(len(X)):
                dis = np.linalg.norm(X[i] - X[j])
                distance.append(dis)
            # index of shortest distances
            nearest = np.array(distance).argsort()
            for num in range(1, self.k+1):
                neighbors_i.append(nearest[num])
            neighbors.append(neighbors_i)
        N = np.array(neighbors)

        # step 2: weights
        n, D = X.shape
        tol = 1e-3
        W = np.zeros((self.k, n))
        I = np.ones((self.k, 1))
        for i in range(len(X)):
            Xi = np.tile(X[i], (self.k, 1)).T
            Ni = X[N[i]].T
            Z = Xi - Ni
            C = np.dot(Z.T, Z)
            C = C + np.identity(self.k) * tol * C.trace()
            C_inv = np.linalg.pinv(C)
            wi = (np.dot(C_inv, I))/(np.dot(np.dot(I.T, C_inv), I)[0, 0])
            W[:, i] = wi[:, 0]

        # step 3: return Y
        W_ = np.zeros((n, n))
        I_ = np.identity(n)
        for i in range(n):
            index = N[i]
            for j in range(self.k):
                W_[index[j], i] = W[j, i]
        M = np.dot((I_ - W_), (I_ - W_).T)
        eig, vec = np.linalg.eig(M)
        index_ = np.argsort(np.abs(eig))[1:self.target_dim + 1]
        Y = vec[:, index_]
        return Y


class KNN(Model):

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, X, y):
        self.data = X
        self.labels = y

    def predict(self, X):
        predict = []
        for i in range(len(X)):
            distance = []
            vote = []
            for j in range(len(self.data)):
                dis = np.linalg.norm(X[i] - self.data[j])
                distance.append(dis)
            # index of shortest distances
            nearest = np.array(distance).argsort()
            for num in range(self.k):
                vote.append(self.labels[nearest[num]])
            predict.append(stats.mode(vote)[0])
        return predict
