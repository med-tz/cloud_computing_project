import numpy as np
import time
import matplotlib.pyplot as plt
from pca_methods.pca_numpy import pca_numpy
from pca_methods.pca_nipal_numpys import pca_niplas_numpy
from pca_methods.pca_nipal_gpu import pca_nipals_gpu


class Excution_time:
    """
    A class used to measure the execution time of different PCA methods
    """
    def __init__(self) -> None:
        """
        Initializes the Excution_time class with necessary PCA method objects and
        creates empty lists to store matrix sizes and execution times.
        """
        self.matrices = []
        self.sizes = []
        self.pca_numpy = pca_numpy()
        self.pca_nipal_numpy = pca_niplas_numpy()
        self.pca_nipal_gpu = pca_nipals_gpu()
        self.times_pca_numpy = []
        self.time_pca_nipal_numpy = []
        self.time_nipal_gpu = []
        self.times_nipal_numpy = []
    def generate_matrices(self):
        """
        Generates a list of random matrices of increasing sizes and stores the matrix
        sizes in a separate list.
        """
        for i in range(100, 5000, 100):
            n, m = i, i
            self.sizes.append(n * m)
            X = np.random.rand(n, m).astype(np.float32)
            self.matrices.append(X)

    def plot_excution_cpu(self):
        """
        Measures the execution time for each matrix in the list using the standard PCA
        and the NIPALS PCA methods implemented in numpy. Plots a graph comparing the 
        execution times of the two methods.
        """
        for X in self.matrices:
            start_time = time.time()
            X_reduced = self.pca_numpy.main(X)
            end_time = time.time()
            execution_time = end_time - start_time
            self.times_pca_numpy.append(execution_time)
            start_time = time.time()
            scores, loading = self.pca_nipal_numpy.main(X)
            end_time = time.time()
            execution_time = end_time - start_time
            self.time_pca_nipal_numpy.append(execution_time)
        plt.figure(figsize=(10, 6))
        plt.plot(self.sizes, self.times_pca_numpy, marker="o", label="PCA (numpy)")
        plt.plot(
            self.sizes, self.time_pca_nipal_numpy, marker="o", label="PCA-NIPAL (numpy)"
        )
        plt.title("Execution time of PCA")
        plt.xlabel("Matrix size")
        plt.ylabel("Execution time (seconds)")
        plt.grid(True)
        plt.legend()
        plt.savefig("results/pca_comparison.png")

    def plot_comparaison_cpu_gpu_nipal(self):
        """
        Measures the execution time for each matrix in the list using the NIPALS PCA
        method implemented in numpy and on the GPU. Plots a graph comparing the 
        execution times of the two methods.
        """
        for X in self.matrices:
            start_time = time.time()
            self.pca_nipal_gpu.main(X)
            end_time = time.time()
            execution_time = end_time - start_time
            self.time_nipal_gpu.append(execution_time)

            start_time = time.time()
            scores, loading = self.pca_nipal_numpy.main(X)
            end_time = time.time()
            execution_time = end_time - start_time
            self.times_nipal_numpy.append(execution_time)
        plt.figure(figsize=(10, 6))
        plt.plot(self.sizes, self.time_nipal_gpu, marker="o", label="PCA-NIPAL (GPU)")
        plt.plot(
            self.sizes, self.times_nipal_numpy, marker="o", label="PCA-NIPAL (numpy)"
        )
        plt.title("Execution time of PCA - NIPAL")
        plt.xlabel("Matrix size")
        plt.ylabel("Execution time (seconds)")
        plt.grid(True)
        plt.legend()
        plt.savefig("pca_comparison_NIPAL.png")


    def main(self):
        """
        Calls the other methods in the class to generate the matrices, 
        measure execution times and plot the results.
        """
        self.generate_matrices()
        self.plot_excution_cpu()
        self.plot_comparaison_cpu_gpu_nipal()


if __name__ == "__main__":
    """
    Creates an object of the Excution_time class and runs the main method.
    """
    cla = Excution_time()
    cla.main()
