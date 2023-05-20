from utils.functions_gpu import Gpu_functions
import random
import numpy as np
import pycuda.gpuarray as gpuarray


class pca_nipals_gpu:
    def __init__(self) -> None:
        """
        Initialize PCA Nipals GPU object with default parameters.
        """
        self.max_iter = 500
        self.num_components = 2
        self.tol = 1e-6
        self.class_object=Gpu_functions()

    def iteration_gpu(self,X,X_gpu,size):
        """
        Perform one iteration of the NIPALS algorithm on GPU.
        
        Parameters:
        X (np.array): Original data.
        X_gpu (gpuarray): Original data stored on GPU.
        size (int): Size of the data array.

        Returns:
        X_gpu (gpuarray): Data after performing one iteration of NIPALS.
        """
        t = X[:, random.randrange(0, size)]
        p = np.zeros((size,)).astype(np.float32)
        t_old = np.zeros_like(t)  
        out_1 = np.zeros_like(t)
        out_2 = np.float32(0)
        output_t = np.zeros(1, dtype=np.float32)
        output_p = np.zeros(1, dtype=np.float32)

        t_gpu = gpuarray.to_gpu(t)
        p_gpu = gpuarray.to_gpu(p)
        t_old_gpu = gpuarray.to_gpu(t_old)
        out_1_gpu = gpuarray.to_gpu(out_1)
        out_2_gpu = gpuarray.to_gpu(np.array([out_2]))
        output_gpu_th = gpuarray.to_gpu(output_t)
        output_gpu_ph = gpuarray.to_gpu(output_p)
        for it in range(self.max_iter):
            out_1_gpu=self.class_object.subtractVectors(t_gpu,t_old_gpu,out_1_gpu,size)
            out_1_cpu = out_1_gpu.get()
            out_2_cpu = np.linalg.norm(out_1_cpu)
            if out_2_cpu < self.tol:
              break
            t_old_gpu = t_gpu.copy()
            self.class_object.multiplyTransposeMatrixVector(X_gpu, t_gpu, p_gpu, size)
            self.class_object.calculateVectorNorm(t_gpu, output_gpu_th, size)
            numpy_array = output_gpu_th.get()
            single_value_0 = float(numpy_array[0])
            self.class_object.divideVectorByNumber(p_gpu, single_value_0, size)
            sum_ph = self.class_object.calculateVectorNorm(p_gpu, output_gpu_ph, size)
            numpy_array = sum_ph.get()
            single_value_1 = float(numpy_array[0])
            norm2_ph = np.float32(np.sqrt(single_value_1))
            self.class_object.divideVectorByNumber(p_gpu, norm2_ph, size)
            self.class_object.multiplyTransposeMatrixVector(X_gpu, p_gpu, t_gpu, size)
            self.class_object.calculateVectorNorm(p_gpu, output_gpu_ph, size)
            numpy_array = output_gpu_th.get()
            single_value_2 = float(numpy_array[0])
            self.class_object.divideVectorByNumber(t_gpu,single_value_2,size)
            X_gpu = self.class_object.update(X_gpu, t_gpu, p_gpu,size)
        return(X_gpu)
    
    def nipals_gpu(self,X):
        """
        Perform the NIPALS algorithm on GPU for a given number of components.

        Parameters:
        X (np.array): Original data.
        """
        X_gpu = gpuarray.to_gpu(X)
        size = X.shape[0]
        for i in range(self.num_components):
            X_gpu= self.iteration_gpu(X,X_gpu,size)

    def main(self,X):
        """
        Main method to call NIPALS algorithm on GPU.

        Parameters:
        X (np.array): Original data.
        """
        self.nipals_gpu(X)

        


            
