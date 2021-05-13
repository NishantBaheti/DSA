import logging
import numpy as np
from numpy.random import random
from noobalgo import __version__
from typing import Union
from .cost import calculate_mse_cost

__author__ = "Nishant Baheti"
__copyright__ = "Nishant Baheti"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


class LinearRegression:
    def __init__(self, alpha: float = 0.01, iterations: int = 10000):
        """Constructor

        Args:
            alpha (float, optional): learning rate. Defaults to 0.01.
            iterations (int, optional): number of iteratons. Defaults to 10000.
        """
        self.alpha = alpha
        self.iterations = iterations
        self._theta = None
        self._X = None
        self._y = None
        self._theta_history = None
        self._cost_history = None

    def _format_x_for_theta_0(self, x_i: np.ndarray) -> np.ndarray:
        """format X matrix for linear model

        put 1's in the first column as feature for theta_0

        Args:
            x_i (np.ndarray): input x matrix

        Returns:
            np.ndarray: formatted x matrix
        """
        x_i = x_i.copy()
        if len(x_i.shape) == 1:
            x_i = x_i.reshape(-1, 1)

        if False in (x_i[..., 0] == 1):
            return np.hstack(tup=(np.ones(shape=(x_i.shape[0], 1)), x_i))
        else:
            return x_i

    @property
    def X(self) -> Union[np.ndarray, None]:
        """property X

        Returns:
            Union[np.ndarray, None]: X matrix
        """
        return self._X

    @property
    def y(self) -> Union[np.ndarray, None]:
        """property y

        Returns:
            Union[np.ndarray, None]: y matrix
        """
        return self._y

    @property
    def theta(self) -> Union[np.ndarray, None]:
        return self._theta

    @property
    def theta_history(self) -> Union[list, None]:
        return self._theta_history

    @property
    def cost_history(self) -> Union[list, None]:
        return self._cost_history

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self._theta is not None:
            format_x = self._format_x_for_theta_0(X)
            if format_x.shape[1] == self._theta.shape[0]:
                y_pred = format_x @ self._theta  # (m,1) = (m,n) * (n,1)
                return y_pred
            elif format_x.shape[1] == self._theta.shape[1]:
                y_pred = format_x @ self._theta.T  # (m,1) = (m,n) * (n,1)
                return y_pred
            else:
                raise ValueError("Shape is not proper.")
        else:
            raise Warning("Model is not trained yet. Theta is None.")

    def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            verbose: bool = True,
            method: str = "SGD",
            theta_precision: float = 0.001,
            batch_size: int = 30
    ) -> None:

        self._X = self._format_x_for_theta_0(X)
        self._y = y

        # number of features+1 because of theta_0
        self._n = self._X.shape[1]
        self._m = self._y.shape[0]

        self._theta_history = []
        self._cost_history = []

        if method == "BGD":
            self._theta = np.random.rand(1, self._n) * theta_precision
            if verbose:
                print("random initial θ value :", self._theta)

            for iteration in range(self.iterations):
                # calculate y_pred
                y_pred = self.predict(self._X)

                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                gradient = np.mean((y_pred - self._y) * self._X, axis=0)
                new_theta = self._theta - (self.alpha * gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_mse_cost(y_pred=self.predict(X=self._X), y=self._y)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        elif method == "SGD":  # stochastic gradient descent
            self._theta = np.random.rand(1, self._n) * theta_precision
            if verbose:
                print("random initial θ value :", self._theta)

            for iteration in range(self.iterations):

                # creating indices for batches
                indices = np.random.randint(0, self._m, size=batch_size)

                # creating batch for this iteration
                X_batch = np.take(self._X, indices, axis=0)
                y_batch = np.take(self._y, indices, axis=0)

                # calculate y_pred
                y_pred = self.predict(X_batch)
                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                gradient = np.mean((y_pred - y_batch) * X_batch, axis=0)
                new_theta = self._theta - (self.alpha * gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_mse_cost(y_pred=self.predict(X=X_batch), y=y_batch)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        elif method == "NORMAL":
            self._theta = np.linalg.inv(self._X.T @ self._X) @ self._X.T @ self._y

        else:
            print("No Method Defined.")


class RidgeRegression:
    def __init__(self, alpha: float = 0.01, iterations: int = 10000):
        self.alpha = alpha
        self.iterations = iterations
        self._theta = None
        self._X = None
        self._y = None
        self._theta_history = None
        self._cost_history = None

    def _format_x_for_theta_0(self, x_i: np.ndarray) -> np.ndarray:
        x_i = x_i.copy()
        if len(x_i.shape) == 1:
            x_i = x_i.reshape(-1, 1)

        if False in (x_i[..., 0] == 1):
            return np.hstack(tup=(np.ones(shape=(x_i.shape[0], 1)), x_i))
        else:
            return x_i

    @property
    def X(self) -> Union[np.ndarray, None]:
        return self._X

    @property
    def y(self) -> Union[np.ndarray, None]:
        return self._y

    @property
    def theta(self) -> Union[np.ndarray, None]:
        return self._theta

    @property
    def theta_history(self) -> Union[list, None]:
        return self._theta_history

    @property
    def cost_history(self) -> Union[list, None]:
        return self._cost_history

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self._theta is not None:
            format_x = self._format_x_for_theta_0(X)

            if format_x.shape[1] == self._theta.shape[0]:
                y_pred = format_x @ self._theta  # (m,1) = (m,n) * (n,1)
                return y_pred
            elif format_x.shape[1] == self._theta.shape[1]:
                y_pred = format_x @ self._theta.T  # (m,1) = (m,n) * (n,1)
                return y_pred
            else:
                raise ValueError("Shape is not proper.")
        else:
            raise Warning("Model is not trained yet. Theta is None.")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        method: str = "SGD",
        theta_precision: float = 0.001,
        penalty: Union[float, int] = 1.0,
        batch_size: int = 30
    ) -> None:

        self._X = self._format_x_for_theta_0(X)
        self._y = y

        # number of features+1 because of theta_0
        self._n = self._X.shape[1]
        self._m = self._y.shape[0]

        self._theta_history = []
        self._cost_history = []

        if method == "BGD":
            self._theta = np.random.rand(1, self._n) * theta_precision
            if verbose:
                print("random initial θ value :", self._theta)

            for iteration in range(self.iterations):
                # calculate y_pred
                y_pred = self.predict(self._X)
                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                ################################################################################################################
                # little bit stretched out
                # new_theta = theta - (alpha * np.sum( ( y_pred - y ) * X, axis = 0 ) * (1 / m)) -  (penalty * theta * (1 / m) )

                gradient = np.mean((y_pred - self._y) * self._X, axis=0)
                new_theta = self._theta * \
                    (1 - (penalty / self._m)) - (self.alpha * gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_mse_cost(y_pred=self.predict(X=self._X), y=self._y)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])
        elif method == "SGD":
            self._theta = np.random.rand(1, self._n) * theta_precision
            if verbose:
                print("random initial θ value :", self._theta)

            for iteration in range(self.iterations):

                indices = np.random.randint(0, self._m, size=batch_size)

                X_batch = np.take(self._X, indices, axis=0)
                y_batch = np.take(self._y, indices, axis=0)

                # calculate y_pred
                y_pred = self.predict(X_batch)
                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                gradient = np.mean((y_pred - y_batch) * X_batch, axis=0)
                new_theta = self._theta * \
                    (1 - (penalty / self._m)) - (self.alpha * gradient)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_mse_cost(y_pred=self.predict(X=X_batch), y=y_batch)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        elif method == "NORMAL":
            self._theta = np.linalg.inv(
                self._X.T @ self._X + (penalty * np.identity(self._n))) @ self._X.T @ self._y

        else:
            print("No Method Defined.")


class LassoRegression:
    def __init__(self, alpha: float = 0.01, iterations: int = 10000):
        self.alpha = alpha
        self.iterations = iterations
        self._theta = None
        self._X = None
        self._y = None
        self._theta_history = None
        self._cost_history = None

    def _format_x_for_theta_0(self, x_i: np.ndarray) -> np.ndarray:
        x_i = x_i.copy()
        if len(x_i.shape) == 1:
            x_i = x_i.reshape(-1, 1)

        if False in (x_i[..., 0] == 1):
            return np.hstack(tup=(np.ones(shape=(x_i.shape[0], 1)), x_i))
        else:
            return x_i

    @property
    def X(self) -> Union[np.ndarray, None]:
        return self._X

    @property
    def y(self) -> Union[np.ndarray, None]:
        return self._y

    @property
    def theta(self) -> Union[np.ndarray, None]:
        return self._theta

    @property
    def theta_history(self) -> Union[list, None]:
        return self._theta_history

    @property
    def cost_history(self) -> Union[list, None]:
        return self._cost_history

    def predict(self, X):
        if self._theta is not None:
            format_x = self._format_x_for_theta_0(X)

            if format_x.shape[1] == self._theta.shape[0]:
                y_pred = format_x @ self._theta  # (m,1) = (m,n) * (n,1)
                return y_pred
            elif format_x.shape[1] == self._theta.shape[1]:
                y_pred = format_x @ self._theta.T  # (m,1) = (m,n) * (n,1)
                return y_pred
            else:
                raise ValueError("Shape is not proper.")
        else:
            raise Warning("model is not trained yet.theta is None.")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        method: str = "SGD",
        theta_precision: float = 0.001,
        penalty: Union[int, float] = 1.0,
        batch_size: int = 30
    ) -> None:

        self._X = self._format_x_for_theta_0(X)
        self._y = y

        # number of features+1 because of theta_0
        self._n = self._X.shape[1]
        self._m = self._y.shape[0]

        self._theta_history = []
        self._cost_history = []

        if method == "BGD":
            self._theta = np.random.rand(1, self._n) * theta_precision
            if verbose:
                print("random initial θ value :", self._theta)

            for iteration in range(self.iterations):
                # calculate y_pred
                y_pred = self.predict(self._X)
                # new θ to replace old θ
                new_theta = np.zeros_like(self._theta)

                # simultaneous operation
                ################################################################################################################
                # little bit stretched out
                # new_theta = theta - (alpha * np.sum( ( y_pred - y ) * X, axis = 0 ) * (1 / m)) -  (penalty * (1 / m) )

                gradient = np.mean((y_pred - self._y) * self._X, axis=0)
                new_theta = self._theta - (self.alpha * gradient) - (penalty/self._m)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_mse_cost(y_pred=self.predict(X=self._X), y=self._y)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        elif method == "SGD":
            self._theta = np.random.rand(1, self._n) * theta_precision
            if verbose:
                print("random initial θ value :", self._theta)

            for iteration in range(self.iterations):

                indices = np.random.randint(0, self._m, size=batch_size)

                X_batch = np.take(self._X, indices, axis=0)
                y_batch = np.take(self._y, indices, axis=0)

                # calculate y_pred
                y_pred = self.predict(X_batch)
                # new θ to replace old θ
                new_theta = None

                # simultaneous operation
                ################################################################################################################
                # little bit stretched out
                # new_theta = theta - (alpha * np.sum( ( y_pred - y ) * X, axis = 0 ) * (1 / m)) -  (penalty * (1 / m) )

                gradient = np.mean((y_pred - y_batch) * X_batch, axis=0)
                new_theta = self._theta - (self.alpha * gradient) - (penalty/self._m)

                if np.isnan(np.sum(new_theta)) or np.isinf(np.sum(new_theta)):
                    print("breaking. found inf or nan.")
                    break
                # override with new θ
                self._theta = new_theta

                # calculate cost to put in history
                cost = calculate_mse_cost(y_pred=self.predict(X=X_batch), y=y_batch)
                self._cost_history.append(cost)

                # calcualted theta in history
                self._theta_history.append(self._theta[0])

        else:
            print("No Method Defined.")
