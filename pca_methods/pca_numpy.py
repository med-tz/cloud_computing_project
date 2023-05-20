import numpy as np


class pca_numpy:
    def __init__(self) -> None:
        """
        Initialize PCA object with default parameters.
        """
        self.num_components = 2

    def standardize_data(self, X):
        """
        Standardize the data by subtracting the mean and dividing by the standard deviation.

        Parameters:
        X (np.array): The data to be standardized.

        Returns:
        X_std (np.array): The standardized data.
        """
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return X_std

    def perform_pca(self, X_std, num_components):
        """
        Perform Principal Component Analysis (PCA) on the standardized data.

        Parameters:
        X_std (np.array): The standardized data.
        num_components (int): The number of principal components to compute.

        Returns:
        X_reduced (np.array): The reduced dataset with the principal components.
        """
        covariance_matrix = np.cov(X_std.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
        X_reduced = np.dot(eigenvector_subset.T, X_std.T).T
        return X_reduced

    def main(self, X):
        """
        Main method to standardize the data and perform PCA.

        Parameters:
        X (np.array): Original data.

        Returns:
        X_reduced (np.array): The reduced dataset with the principal components.
        """
        X_std = self.standardize_data(X)
        X_reduced = self.perform_pca(X_std, self.num_components)
        return X_reduced
