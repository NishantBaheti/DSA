
import logging
import numpy as np 
from noobalgo import __version__

__author__ = "Nishant Baheti"
__copyright__ = "Nishant Baheti"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

class LinearRegression:
    def __init__(self,alpha = 0.01 ,iterations = 10000):
        self.alpha = alpha
        self.iterations = iterations
        self._theta = None
        self._X = None
        self._y = None
        self._theta_history = None
        self._cost_history = None

    def _format_X_for_theta_0(self,X_i):

        X_i = X_i.copy()
        if len(X_i.shape) == 1:
            X_i = X_i.reshape(-1,1)

        if False in (X_i[...,0] == 1):
            return np.hstack(tup=(np.ones(shape=(X_i.shape[0],1)) , X_i))
        else:
            return X_i

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def theta(self):
        return self._theta

    @property
    def theta_history(self):
        return self._theta_history

    @property
    def cost_history(self):
        return self._cost_history

    def predict(self,X):
        format_X = self._format_X_for_theta_0(X)

        if format_X.shape[1] == self._theta.shape[0]:
            y_pred = format_X @ self._theta # (m,1) = (m,n) * (n,1)
            return y_pred
        elif format_X.shape[1] == self._theta.shape[1]:
            y_pred = format_X @ self._theta.T # (m,1) = (m,n) * (n,1)
            return y_pred
        else:
            raise ValueError("Shape is not proper.")


    def train(self, X, y, verbose=True, method="BGD", theta_precision = 0.001, batch_size=30):

        self._X = self._format_X_for_theta_0(X)
        self._y = y

        # number of features+1 because of theta_0
        self._n = self._X.shape[1]
        self._m = self._y.shape[0]

        self._theta_history = []
        self._cost_history = []

        if method == "BGD":
            self._theta = np.random.rand(1,self._n) * theta_precision
            if verbose: print("random initial θ value :",self._theta)

            for iteration in range(self.iterations):
                # calculate y_pred
                y_pred = self.predict(self._X)

                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                gradient = np.mean( ( y_pred - self._y ) * self._X, axis = 0 )
                new_theta = self._theta - (self.alpha *  gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_cost(y_pred = self.predict(X=self._X), y = self._y)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        elif method == "SGD": # stochastic gradient descent
            self._theta = np.random.rand(1,self._n) * theta_precision
            if verbose: print("random initial θ value :",self._theta)

            for iteration in range(self.iterations):

                # creating indices for batches
                indices = np.random.randint(0,self._m,size=batch_size)

                # creating batch for this iteration
                X_batch = np.take(self._X,indices,axis=0)
                y_batch = np.take(self._y,indices,axis=0)

                # calculate y_pred
                y_pred = self.predict(X_batch)
                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                gradient = np.mean( ( y_pred - y_batch ) * X_batch, axis = 0 )
                new_theta = self._theta - (self.alpha *  gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_cost(y_pred = self.predict(X=X_batch), y = y_batch)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        elif method == "NORMAL":
            self._theta = np.linalg.inv(self._X.T @ self._X) @ self._X.T @ self._y

        else:
            print("No Method Defined.")

class RidgeRegression:
    def __init__(self,alpha = 0.01 ,iterations = 10000):
        self.alpha = alpha
        self.iterations = iterations
        self._theta = None
        self._X = None
        self._y = None
        self._theta_history = None
        self._cost_history = None

    def _format_X_for_theta_0(self,X_i):

        X_i = X_i.copy()
        if len(X_i.shape) == 1:
            X_i = X_i.reshape(-1,1)

        if False in (X_i[...,0] == 1):
            return np.hstack(tup=(np.ones(shape=(X_i.shape[0],1)) , X_i))
        else:
            return X_i

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def theta(self):
        return self._theta

    @property
    def theta_history(self):
        return self._theta_history

    @property
    def cost_history(self):
        return self._cost_history

    def predict(self,X):
        format_X = self._format_X_for_theta_0(X)

        if format_X.shape[1] == self._theta.shape[0]:
            y_pred = format_X @ self._theta # (m,1) = (m,n) * (n,1)
            return y_pred
        elif format_X.shape[1] == self._theta.shape[1]:
            y_pred = format_X @ self._theta.T # (m,1) = (m,n) * (n,1)
            return y_pred
        else:
            raise ValueError("Shape is not proper.")

    def train(self, X, y, verbose=True, method="BGD", theta_precision = 0.001, penalty=1.0, batch_size=30):

        self._X = self._format_X_for_theta_0(X)
        self._y = y

        # number of features+1 because of theta_0
        self._n = self._X.shape[1]
        self._m = self._y.shape[0]

        self._theta_history = []
        self._cost_history = []

        if method == "BGD":
            self._theta = np.random.rand(1,self._n) * theta_precision
            if verbose: print("random initial θ value :",self._theta)

            for iteration in range(self.iterations):
                # calculate y_pred
                y_pred = self.predict(self._X)
                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                ################################################################################################################
                # little bit stretched out
                # new_theta = theta - (alpha * np.sum( ( y_pred - y ) * X, axis = 0 ) * (1 / m)) -  (penalty * theta * (1 / m) )

                gradient = np.mean( ( y_pred - self._y ) * self._X, axis = 0 )
                new_theta = self._theta * (1 - (penalty/self._m) ) - (self.alpha * gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_cost(y_pred = self.predict(X=self._X), y = self._y)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])
        elif method == "SGD":
            self._theta = np.random.rand(1,self._n) * theta_precision
            if verbose: print("random initial θ value :",self._theta)

            for iteration in range(self.iterations):

                indices = np.random.randint(0,self._m,size=batch_size)

                X_batch = np.take(self._X,indices,axis=0)
                y_batch = np.take(self._y,indices,axis=0)

                # calculate y_pred
                y_pred = self.predict(X_batch)
                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                gradient = np.mean( ( y_pred - y_batch) * X_batch, axis = 0 )
                new_theta = self._theta * (1 - (penalty/self._m) ) - (self.alpha * gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_cost(y_pred = self.predict(X=X_batch), y = y_batch)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        elif method == "NORMAL":
            self._theta = np.linalg.inv(self._X.T @ self._X + (penalty * np.identity(self._n))) @ self._X.T @ self._y

        else:
            print("No Method Defined.")

