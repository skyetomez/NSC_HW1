import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
import matplotlib.pyplot as plt


############ PCA ############

class PCA:
    """ PCA for question number 5"""

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None
        self.eigenvalues = None

    def fit(self, X):
        time, height, width = X.shape
        X = X.reshape(time, height * width)
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X)  # time covariance matrix
        self.eigenvalues, self.eigenvectors = la.eig(cov)
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

    def transform(self, X):
        return (X - self.mean) @ self.eigenvectors[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def skree_plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        ax.bar(self.eigenvalues)
        ax.title('Scree Plot')
        ax.xlabel('Principal Component')
        ax.ylabel('Eigenvalue')
        ax.show()
