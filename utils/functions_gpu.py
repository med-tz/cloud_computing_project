import pycuda.driver as cuda
from pycuda import driver, compiler
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

class Gpu_functions:
    """
    Class to perform common operations on GPU using PyCUDA.
    """
    def __init__(self) -> None:
        """Initializes the maximum block dimension attribute.
            Attributes: MAX_BLOCK_DIM (int): Maximum block dimension allowed in the GPU.
        """
        self.MAX_BLOCK_DIM = 1024
    
    def subtractVectors(self, vector1_gpu, vector2_gpu, result_gpu, size):
        """
        Subtract two vectors on the GPU.

        Args:
            vector1_gpu (gpuarray): First input vector stored on the GPU.
            vector2_gpu (gpuarray): Second input vector stored on the GPU.
            result_gpu (gpuarray): Result vector stored on the GPU.
            size (int): Size of the vectors.

        Returns:
            result_gpu (gpuarray): The resulting vector after subtraction stored on the GPU.
        """
        subtractKernel = SourceModule("""
            #define SIZE %d
            __global__ void subtractVectors(float *vector1, float *vector2, float *resultantVector) {
                int index = threadIdx.x + blockIdx.x * blockDim.x;

                if(index < SIZE) {
                    resultantVector[index] = vector1[index] - vector2[index];
                }
            }
            """ % size)

        gpu_subtract = subtractKernel.get_function("subtractVectors")
        block_size = (min(size, self.MAX_BLOCK_DIM), 1, 1)
        grid_size = ((size + block_size[0] - 1) // block_size[0], 1, 1)
        
        gpu_subtract(vector1_gpu, vector2_gpu, result_gpu, block=block_size, grid=grid_size)

        return result_gpu

    def multiplyMatrixVector(self, Matrrix_gpu, vector_gpu, result_gpu, size):
        """
        Multiply a matrix with a vector on the GPU.

        Args:
            Matrrix_gpu (gpuarray): Input matrix stored on the GPU.
            vector_gpu (gpuarray): Input vector stored on the GPU.
            result_gpu (gpuarray): Result vector stored on the GPU.
            size (int): Size of the matrix and vector.

        Returns:
            result_gpu (gpuarray): The resulting vector after multiplication stored on the GPU.
        """
        multiplyMatrixVector = SourceModule("""
                #define SIZE %d

                __global__ void multiplyMatrixWithVector(float *matrix_multiply, float *Vector_multiply, float *resultantVector_multiply) {
                    int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
                    
                    if(rowIndex < SIZE) {
                        float tempSum_multiply = 0.0f;
                        for (int i = 0; i < SIZE; ++i) {   
                            tempSum_multiply += matrix_multiply[i + rowIndex * SIZE] * Vector_multiply[i];     
                        }

                        resultantVector_multiply[rowIndex] = tempSum_multiply;
                    }
                }
                """ % size)

        gpu_multiply = multiplyMatrixVector.get_function("multiplyMatrixWithVector")
        block_size = (min(size, self.MAX_BLOCK_DIM), 1, 1)
        grid_size = ((size + block_size[0] - 1) // block_size[0], 1, 1)
        gpu_multiply(Matrrix_gpu, vector_gpu, result_gpu,block=block_size, grid=grid_size)
        return result_gpu

    def multiplyTransposeMatrixVector(self, Matrrix_gpu, vector_gpu, result_gpu, size):
        """
        Multiply a transpose matrix with a vector on the GPU.

        Args:
            Matrrix_gpu (gpuarray): Input matrix stored on the GPU.
            vector_gpu (gpuarray): Input vector stored on the GPU.
            result_gpu (gpuarray): Result vector stored on the GPU.
            size (int): Size of the matrix and vector.

        Returns:
            result_gpu (gpuarray): The resulting vector after multiplication stored on the GPU.
        """
        multiplyTransposeMatrixVector = SourceModule("""
                #define SIZE %d

                __global__ void multiplyTransposeMatrixWithVector(float *matrix, float *Vector, float *resultantVector) {
                    int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;

                    float tempSum = 0.0f;
                    for (int i = 0; i < SIZE; ++i) {   
                        tempSum += matrix[rowIndex + i * SIZE] * Vector[i];     
                    }

                    resultantVector[rowIndex] = tempSum;
                }
                """ % size)

        gpu_multiply = multiplyTransposeMatrixVector.get_function("multiplyTransposeMatrixWithVector")
        block_size = (min(size, self.MAX_BLOCK_DIM), 1, 1)
        grid_size = ((size + block_size[0] - 1) // block_size[0], 1, 1)
        gpu_multiply(Matrrix_gpu, vector_gpu, result_gpu,block=block_size, grid=grid_size)
        return result_gpu
    
    def calculateVectorNorm(self, vector_gpu, result_gpu, size):
        """
        Calculate the norm of a vector on the GPU.

        Args:
            vector_gpu (gpuarray): Input vector stored on the GPU.
            result_gpu (gpuarray): Result vector stored on the GPU.
            size (int): Size of the vector.

        Returns:
            result_gpu (gpuarray): The norm of the vector stored on the GPU.
        """
        calculateVectorNorm = SourceModule( """
            #define SIZE %(size)d
            #define MAX_BLOCK_DIM %(MAX_BLOCK_DIM)d

            __global__ void calculateVectorNorm(float *vector, float *result) {
                __shared__ float squaredElements[MAX_BLOCK_DIM];
                __shared__ float blockSize;

                int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;

                // Squaring the elements of the vector
                squaredElements[threadIdx.x] = vector[globalThreadIndex] * vector[globalThreadIndex];

                // Calculate the size of the current block
                if (threadIdx.x == 0) {
                    if ((blockIdx.x + 1) * MAX_BLOCK_DIM <= SIZE) {
                        blockSize = MAX_BLOCK_DIM;
                    } else {
                        blockSize = fmodf(SIZE, MAX_BLOCK_DIM);
                    }
                }

                __syncthreads();

                // Sum up the squared elements in the block
                if (threadIdx.x == 0) {
                    float sumOfSquaredElements = 0.0f;
                    for (int i = 0; i < blockSize; i++) {
                        sumOfSquaredElements += squaredElements[i];
                    }

                    // Atomic add to prevent race conditions when updating the result
                    atomicAdd(result, sumOfSquaredElements);
                }
            }
            """ % {"size": size, "MAX_BLOCK_DIM": self.MAX_BLOCK_DIM})

        calculateVectorNorm = calculateVectorNorm.get_function("calculateVectorNorm")
        blocksPerGrid = (int((size + self.MAX_BLOCK_DIM-1)/self.MAX_BLOCK_DIM), 1)
        calculateVectorNorm(vector_gpu, result_gpu, block=(self.MAX_BLOCK_DIM, 1, 1), grid=blocksPerGrid)

        return result_gpu
    
    def divideVectorByNumber(self, vector_gpu, divisor, size):
        """
        Divide a vector by a number on the GPU.

        Args:
            vector_gpu (gpuarray): Input vector stored on the GPU.
            divisor (float): Number to divide the vector by.
            size (int): Size of the vector.

        Returns:
            vector_gpu (gpuarray): The resulting vector after division stored on the GPU.
        """
        divideVectorByNumberKernel = SourceModule("""
            __global__ void divideVectorByNumber(float *vector, float divisor){
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            vector[index] /= divisor;
            }
            """)

        gpu_divide = divideVectorByNumberKernel.get_function("divideVectorByNumber")
        num_threads = int(np.ceil(size))
        grid_size = int(np.ceil(num_threads / self.MAX_BLOCK_DIM))
        if grid_size > 1:
            block_size = self.MAX_BLOCK_DIM
        else:
            block_size = num_threads

        # Launch the kernel
        gpu_divide(vector_gpu, np.float32(divisor), block=(block_size, 1, 1), grid=(grid_size, 1, 1))

        return vector_gpu
    
    def update(self,X_gpu, th_gpu, ph_gpu, size):
        """
        Update a matrix on the GPU by subtracting the outer product of two vectors from it.

        Args:
            X_gpu (gpuarray): Input matrix stored on the GPU.
            th_gpu (gpuarray): First input vector stored on the GPU.
            ph_gpu (gpuarray): Second input vector stored on the GPU.
            size (int): Size of the matrix and vectors.

        Returns:
            X_gpu (gpuarray): The updated matrix stored on the GPU.
        """
        outer_mult = SourceModule("""
                # include <stdio.h>

                # define size %d
                __global__ void outer_mult(float *X, float *T ,  float *P)
                {
                int bx = blockIdx.x;
                int by = blockIdx.y;
                int tx = threadIdx.x;
                int ty = threadIdx.y;
                // Block row and column
                int row = by*blockDim.y + ty;
                int col = bx*blockDim.x + tx;
                //int dim = gridDim.x*blockDim.x;
                if (row < size && col < size){
                    int idx = row*size + col ;
                    X[idx] -= T[row]*P[col];
                }
                }
                """ % size)
        outer = outer_mult.get_function("outer_mult")
        block_size = (min(size, 32), min(size, 32), 1)
        grid_size = (int(np.ceil(size / block_size[0])), int(np.ceil(size / block_size[1])), 1)
        outer(X_gpu, th_gpu, ph_gpu, block=block_size, grid=grid_size)
        return X_gpu