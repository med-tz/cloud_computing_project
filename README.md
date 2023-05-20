# cloud_computing_project

## Overview
This project is designed to compare different implementations of Principal Component Analysis (PCA). Initially, the standard PCA method is employed as a benchmark for comparison. Then we implement the Nonlinear Iterative Partial Least Squares (NIPALS) algorithm, a variant of PCA, and explore two distinct versions of this algorithm.

The first NIPALS implementation is parallelized on a Graphics Processing Unit (GPU) to leverage the high computational power and speed of GPUs. The second one is executed on a Central Processing Unit (CPU), utilizing the NumPy library for computations. This project allows us to examine the performance differences between various PCA implementations and between different computing paradigms (GPU vs CPU).

## Installation

### Dependencies

The only external library required for this project is PyCUDA. You can install it using pip:

```bash
!pip install pycuda
```

## Usage

1. After installing the necessary dependencies, navigate to the directory where you've cloned or downloaded this repository.
2. Then simply execute the `main.py` script. 

```bash
python main.py
```

The results will be generated and stored in the `results` directory as `.png` files.





