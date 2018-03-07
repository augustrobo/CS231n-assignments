import numpy as np
from scipy.stats import mode

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """
    def __init__(self):
        pass

    def train(self, X, y):
        """
        X: X.shape == (N, D), N examples, each of dim D
        y: y.shape == (N,)
           y[i] is the label of X[i]
        """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtrain = X
        self.ytrain = y

    def computeDistances(self, X):
        """
        Compute the distances between each test point in X
        and each training point in self.Xtrain
        Input:
        X: each row is an example we wish to predict label for
           X.shape == (ntest, D)
        Output:
        dists: dists.shape == (ntest, ntrain)
               dists[i, j] == L2 distance between X[i] and self.Xtrain[j]
        """
        ntest, ntrain = X.shape[0], self.Xtrain.shape[0]

        te = np.sum(X * X, axis = 1, keepdims = True)
        tr = np.sum(self.Xtrain * self.Xtrain, axis = 1, keepdims = True)
        dists = np.sqrt(-2 * np.dot(X, self.Xtrain.T) + te + tr.T)

    def predict(self, X, k = 1):
        """
        Predict labels for test data using this classifier.

        Input:
        X: each row is an example we wish to predict label for
           X.shape == (ntest, D)
        Output:
        ypred: ypred.shape == (ntest,)
               ypred[i] is the predicted label for X[i]
        """
        dists = self.computeDistances(X)
        ntest = X.shape[0]
        # ith row: indices of k nearest neighbors of X[i]
        k_idx = dists.argsort(axis = 1)[:, :k]
        # ith row: labels of k nearest neighbors of X[i]
        closest_y = self.ytrain[k_idx]
        # ith row: most common label in closest_y[i]
        y_pred = mode(closest_y, axis = 1).mode

        return np.squeeze(y_pred)
