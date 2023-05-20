import numpy as np
import random

class pca_niplas_numpy:
    def __init__(self) -> None:
        """
        Initialize PCA NIPALS object with default parameters.
        """
        self.num_components = 2
        self.max_iter = 500
        self.tol = 1e-6

    def nipals(self, X, num_components, max_iter, tol):
        """
        Perform Non-linear Iterative Partial Least Squares (NIPALS) algorithm 
        for Principal Component Analysis (PCA).
        
        Parameters:
        X (np.array): The data to perform PCA on.
        num_components (int): The number of principal components to compute.
        max_iter (int): The maximum number of iterations for the NIPALS algorithm.
        tol (float): The tolerance for the stopping condition.

        Returns:
        scores (np.array): The scores (i.e., the transformed dataset).
        loadings (np.array): The loadings (i.e., the principal components).
        """
        size = X.shape[0]
        loadings = np.zeros((size, num_components))
        scores = np.zeros((size, num_components))

        for i in range(num_components):
            t = X[:,random.randrange(0, size)].reshape(-1, 1)
            t_old = np.zeros_like(t)
            it = 0
            while np.linalg.norm(t - t_old) > tol and it < max_iter:
                t_old = t
                p = np.dot(X.T, t) / (np.linalg.norm(t))**2
                p /= np.linalg.norm(p)
                t = np.dot(X, p) / (np.linalg.norm(p))**2
                it += 1
            loadings[:, i] = p.flatten()
            scores[:, i] = t.flatten()
            X -= np.outer(t, p.T)

        return scores, loadings

    def main(self, X):
        """
        Main method to call NIPALS algorithm.

        Parameters:
        X (np.array): Original data.

        Returns:
        scores, loadings (tuple): A tuple of scores and loadings.
        """
        scores, loadings = self.nipals(X, self.num_components, self.max_iter, self.tol)
        return (scores, loadings)
