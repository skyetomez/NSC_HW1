import jax.numpy as np
import numpy as nnp
import jax.numpy.linalg as la
from tqdm import tqdm
# import numpy as np
# import numpy.linalg as la
from numpy.typing import NDArray


# need to minimize the frobenius norm of V st that WH are nonnegative
"""
Using the Lee and Seung multiplicative update rules, 
implement a function that performs NMF on a given matrix V.
This implementation takes too long to run so falling back on sklearn's version. 
"""


def get_hw5_2_solution(movie, num_components):
    from sklearn.decomposition import NMF
    print("getting solution. This may take awhile...")
    nmf = NMF(n_components=num_components,
              init='random',
              random_state=0,
              solver='mu',
              max_iter=200,
              tol=1e-4,
              verbose=1)
    W = nmf.fit_transform(movie.reshape(movie.shape[0], -1).T)
    return W.reshape(*movie.shape[1:], num_components).T


class NMF:
    # ||V-WH||_f^2
    def __init__(self, num_components) -> None:
        self.num_components = num_components

    def fit(self, V,  max_iter: int = 200, tol: float = 1e-4):
        time, height, width = V.shape
        hw = height * width
       # (time, hw)
        V = V.reshape(time, hw)
        W, H = self.minimization_loop(V, max_iter, tol)
        return W, H

    def fit_transform(self, V, type: str, max_iter: int = 200, tol: float = 1e-4):
        W, H = self.fit(V, max_iter, tol)
        if type == 'full':
            return np.dot(W, H).reshape(*V.shape)
        if type == 'partial':
            return H.reshape(self.num_components, *V.shape[1:])
        else:
            print("Invalid type. Returning partial reconstruction.")
            return H.reshape(self.num_components, *V.shape[1:])

    # ||V-WH||_f^2

    def minimization_loop(self, V, max_iter: int = 200, tol: float = 1e-4):

        W = self._init_random(V, (V.shape[0], self.num_components))
        H = self._init_random(V, (self.num_components, V.shape[1]))

        n_iter = 0
        vareps = 1e-9

        with tqdm(total=max_iter, desc="NMF Progress", leave=False) as pbar:
            for n_iter in range(max_iter):
                # (ncomp, time) @ (time, hw) / (ncomp, hw) @ (hw, ncomp) @ (ncomp, hw)
                H *= np.dot(W.T, V) / (np.dot(np.dot(W.T, W), H) + vareps)
                W *= np.dot(V, H.T) / (np.dot(np.dot(W, H), H.T) + vareps)

                pbar.update(1)

                if n_iter % 10 == 0:
                    cost = self._check_distance(V, W, H)
                    pbar.set_description(
                        f"Iteration {n_iter}: cost={cost:.4f}")
                    if cost < tol:
                        pbar.write(f"Converged in {n_iter} iterations.")
                        break

                n_iter += 1

        print(f"Finished in {max_iter} iterations; May not have converged.")
        return W, H

    def _init_random(self, V: NDArray, shape: tuple = None):
        mu = np.sqrt(V.mean()/self.num_components)
        return np.abs(nnp.random.normal(mu, 1, shape))

    def _check_distance(self, V, W, H):
        yhat = np.dot(W, H)
        return la.norm(V - yhat, 'fro')
