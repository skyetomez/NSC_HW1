import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray


# need to minimize the frobenius norm of V st that WH are nonnegative
"""
Using the Lee and Seung multiplicative update rules, implement a function that performs NMF on a given matrix V.

"""


class NMF:
    # ||V-WH||_f^2
    def __init__(self) -> None:
        self.V = None
        self.W = None
        self.H = None

    def fit(self, V, num_components, tol: float = 1e-6):
        time, height, width = V.shape
        hw = height * width
        V = V.reshape(time, hw)
        W = np.abs(np.random.normal(2, 1, (hw, num_components)))
        H = np.abs(np.random.normal(2, 1, (hw, num_components)))
        self.minimization_loop(W, V, H, tol)
        return W, H

    # ||V-WH||_f^2
    def minimization_loop(self, W, V, H, tol: float = 1e-6):
        while la.norm(H_new-H, 'fro') > tol and la.norm(W_new-W, 'fro') > tol:
            H_new = H * (W.T * V) / (W.T * W * H)
            W_new = W * (V * H_new.T) / (W * H_new * H_new.T)
            H = H_new
            W = W_new
