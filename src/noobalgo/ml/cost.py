from typing import Union
import numpy as np


def calculate_mse_cost(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Calculate error for regression model with mean squared error

    Args:
        y_pred (np.ndarray): predicted y value, y^.
        y (np.ndarray): actual y value. 

    Returns:
        float: mean squared error cost 
    """
    return np.mean(np.square(y_pred - y)) / 2
