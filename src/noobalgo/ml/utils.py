import numpy as np
from typing import Union


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function

    .. math::
        h_\theta(z) = \frac{1}{1 + e^{-z}}

    Args:
        z (ndarray): input value

    Returns:
        np.ndarray: sigmoid value
    """
    return 1 / (1 + np.exp(-z))
