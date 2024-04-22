import numpy as np
import jax.numpy as jnp
import jax.numpy.linalg as la
from numpy.typing import NDArray


########### Independent Component Analysis ############
# https://en.wikipedia.org/wiki/FastICA

class ICA:
    def __init__(self, n_components: int = 1) -> None:
        self.n_components = n_components

    def fit(self, X: NDArray, max_iter: int = 200, tol: float = 1e-4) -> NDArray:
        self._pre_process(X)
        for idx in range(self.n_components):
            w = self.W[:, idx]
            for _ in range(max_iter):
                w_plus = self._update(w, X)
                if idx > 1:
                    w_plus = self._orthogonalize(w_plus, idx)
                w_plus /= la.norm(w_plus, 2)
                if la.norm(w_plus - w) < tol:
                    break
            self.W[:, idx] = w_plus
        return self.W

    def fit_transform(self, X: NDArray) -> NDArray:
        W = self.fit(X)
        return jnp.dot(W.T, X)

    def _pre_process(self, X: NDArray) -> None:
        self._X_reshaped = X.reshape(X.shape[0], -1)
        self._X_reshaped = self._whiten(self._X_reshaped)
        self._init_random()

    def _init_random(self) -> None:
        t, _ = self._X_reshaped.shape
        self.W = np.random.normal(size=(t, self.n_components))
        self.W /= self.W.mean(axis=0)

    def _update(self, w: NDArray, X: NDArray) -> NDArray:
        term1 = jnp.dot(X, self._g(jnp.dot(w.T, X)).T)
        term2 = jnp.multiply(self._g_prime(
            jnp.sum(jnp.dot(w.T, X))), jnp.sum(w))
        return term1 - term2

    def _orthogonalize(self, w: NDArray, idx: int) -> NDArray:
        inner_product = jnp.multiply(self.W[:, idx], w)
        inner_product = jnp.sum(inner_product, axis=0)
        return w - jnp.sum(inner_product)

    def _whiten(self, X: NDArray) -> NDArray:
        cov = jnp.cov(X)
        D, E = la.eig(cov)
        D_sqrt = jnp.diag(jnp.sqrt(D))
        return jnp.dot(jnp.dot(D_sqrt, E.T), X)

    def _f(self, X: NDArray) -> NDArray:
        # return jnp.log(jnp.cosh(X))
        return -jnp.exp(-0.5*jnp.square(X))

    def _g(self, X: NDArray) -> NDArray:
        # return jnp.tanh(X)
        return -jnp.multiply(X, jnp.exp(-0.5*jnp.square(X)))

    def _g_prime(self, X: NDArray) -> NDArray:
        # return 1 - jnp.square(jnp.tanh(X))
        return jnp.multiply((1 - jnp.square(X)), jnp.exp(-0.5*jnp.square(X)))
