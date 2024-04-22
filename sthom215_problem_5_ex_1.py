import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
import matplotlib.pyplot as plt


############ PCA ############

class PCA:
    """ PCA for question number 5"""

    __slots__ = ['n_components',
                 'mean',
                 'eigenvectors',
                 'eigenvalues',
                 'cov',
                 '_X',
                 '_X_centered',
                 'time',
                 'height',
                 'width']

    def __init__(self, n_components: int = -1):
        self.n_components = n_components
        self.mean: NDArray[np.float64] = None
        self.eigenvectors: NDArray[np.float64] = None
        self.eigenvalues: NDArray[np.float64] = None
        self.cov: NDArray[np.float64] = None

    def fit(self, X: NDArray[np.float64]) -> None:
        self.time, self.height, self.width = X.shape
        self._X = X.reshape(self.time, self.height * self.width)
        self.mean = np.mean(X, axis=0)
        self._X_centered = (X - self.mean).reshape(self.time,
                                                   self.height * self.width)
        self.cov = np.cov(self._X_centered)  # time covariance matrix
        self.eigenvalues, self.eigenvectors = la.eig(self.cov)
        if self.n_components == -1:
            self.n_components = len(self.eigenvalues)
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Must be run after fit method."""
        if self.n_components == -1:
            self.n_components = len(self.eigenvalues)
            return self.eigenvectors.T @ self._X_centered
        return self.eigenvectors[:, :self.n_components].T @ self._X_centered

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self.fit(X)
        transformed = self.transform(X)
        return transformed.reshape(self.n_components, self.height, self.width)

    def skree_plot(self) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        ax.plot(self.eigenvalues,  color='black',
                marker='o', linestyle='dashed', linewidth=2)
        ax.set_title('Scree Plot')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Eigenvalue')
        fig.show()
