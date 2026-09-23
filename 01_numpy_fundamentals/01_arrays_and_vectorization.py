"""
=============================================================================
LESSON 02: NUMPY ESSENTIALS — VECTORS, MATRICES & VECTORIZATION
=============================================================================
Why NumPy?
In Machine Learning, almost everything is represented as numbers:
- An image is a 3D/4D grid of numbers (Height x Width x Channels)
- Text is converted into vectors of numbers (Embeddings)
- A dataset is a 2D table (Samples x Features)
- Neural network weights are matrices of numbers

Pure Python lists are too slow and consume too much memory for millions
of calculations. NumPy provides C-speed, contiguous memory arrays (ndarrays)
and vectorized math.

Run this file with:
    python 01_numpy_fundamentals/01_arrays_and_vectorization.py
=============================================================================
"""

import time
import numpy as np


def separator(title: str):
    print(f"\n{'=' * 22} {title} {'=' * 22}")


# ---------------------------------------------------------------------------
# 1. WHY NUMPY? SPEED & VECTORIZATION (BENCHMARK)
# ---------------------------------------------------------------------------
separator("1. Why NumPy? (Speed Benchmark)")

SIZE = 1_000_000

# Regular Python List
list_a = list(range(SIZE))
list_b = list(range(SIZE))

start = time.perf_counter()
# Standard python loop to add elements
list_result = [a + b for a, b in zip(list_a, list_b)]
py_time = time.perf_counter() - start
print(f"Python list comprehension ({SIZE:,} items): {py_time:.4f} seconds")

# NumPy Array
np_a = np.arange(SIZE)
np_b = np.arange(SIZE)

start = time.perf_counter()
# Vectorized addition — executed at compiled C-level in parallel!
np_result = np_a + np_b
np_time = time.perf_counter() - start
print(f"NumPy vectorized addition ({SIZE:,} items):    {np_time:.4f} seconds")
print(f"--> NumPy is ~{py_time / np_time:.1f}x faster!\n")


# ---------------------------------------------------------------------------
# 2. CREATING ARRAYS & UNDERSTANDING SHAPES (TENSORS)
# ---------------------------------------------------------------------------
separator("2. Creating Arrays & Tensor Dimensions")

# 1D Array (Vector) — e.g. a single sample's features or bias vector
vector = np.array([1.5, 2.7, 3.1, 4.0])
print("1D Vector:", vector)
print(f"  Shape: {vector.shape}, Dimensions (ndim): {vector.ndim}, Data Type (dtype): {vector.dtype}")

# 2D Array (Matrix) — e.g. a batch of samples or weight matrix
# Shape: (rows, columns) -> (samples, features)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("\n2D Matrix:\n", matrix)
print(f"  Shape: {matrix.shape} (2 rows, 3 columns), Dimensions: {matrix.ndim}")

# Handy initializers used constantly in ML:
zeros = np.zeros((2, 4))             # Weights initialization (2 rows, 4 columns)
ones = np.ones((3, 3))               # All ones
linear_space = np.linspace(0, 1, 5)  # 5 evenly spaced numbers from 0.0 to 1.0
random_normal = np.random.randn(2, 3) # Random numbers from Standard Normal (mean 0, variance 1)

print("\nnp.zeros((2, 4)):\n", zeros)
print("np.linspace(0, 1, 5):", linear_space)
print("np.random.randn(2, 3) (e.g. initial weights):\n", np.round(random_normal, 3))


# ---------------------------------------------------------------------------
# 3. RESHAPING & FLATTENING (CRITICAL FOR NEURAL NETWORKS)
# ---------------------------------------------------------------------------
separator("3. Reshaping Arrays")

# Suppose you have 12 values:
arr = np.arange(12)  # [0, 1, 2, ..., 11]
print("Original 1D array:", arr)

# Reshape to (3, 4) — 3 rows, 4 columns
matrix_3x4 = arr.reshape((3, 4))
print("\nReshaped to (3, 4):\n", matrix_3x4)

# Use -1 to let NumPy infer the dimension automatically:
# If we want 2 rows, NumPy calculates 12 / 2 = 6 columns:
matrix_2x6 = arr.reshape((2, -1))
print("\nReshaped with -1 (2, -1) -> inferred shape:", matrix_2x6.shape)

# Flattening (e.g. converting a 2D image matrix back to a 1D vector before dense layers):
flat = matrix_3x4.flatten()
print("Flattened back to 1D:", flat.shape)


# ---------------------------------------------------------------------------
# 4. ELEMENT-WISE MATH VS MATRIX MULTIPLICATION (DOT PRODUCT)
# ---------------------------------------------------------------------------
separator("4. Element-wise vs Matrix Multiplication (@)")

X = np.array([[1, 2],
              [3, 4]])

W = np.array([[5, 6],
              [7, 8]])

# Element-wise operations (operators: +, -, *, /, **)
print("Element-wise multiplication (X * W):\n", X * W)
# 1*5=5, 2*6=12, 3*7=21, 4*8=32

# Dot Product / Matrix Multiplication (The core operation of Deep Learning: Y = X @ W)
# Using the `@` operator or np.dot()
# Row 0 of X dot Column 0 of W: (1*5 + 2*7) = 19
# Row 0 of X dot Column 1 of W: (1*6 + 2*8) = 22
# Row 1 of X dot Column 0 of W: (3*5 + 4*7) = 43
# Row 1 of X dot Column 1 of W: (3*6 + 4*8) = 50
print("\nMatrix multiplication (X @ W) [Linear Layer / Dot Product]:\n", X @ W)


# ---------------------------------------------------------------------------
# 5. BROADCASTING (HOW NUMPY HANDLES DIFFERENT SHAPES)
# ---------------------------------------------------------------------------
separator("5. Broadcasting Rules")
# In ML, we often add a 1D bias vector to a 2D batch of samples.
# NumPy automatically 'broadcasts' the smaller dimension to match the larger one!

samples = np.array([
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0]
])  # Shape (2, 3)

bias = np.array([1.0, 2.0, 3.0])  # Shape (3,)

# The bias is automatically broadcasted across all rows:
output = samples + bias
print("Batch (2, 3):\n", samples)
print("Bias (3,):\n", bias)
print("Result of (samples + bias):\n", output)


# ---------------------------------------------------------------------------
# 6. SLICING, INDEXING & BOOLEAN MASKING (FILTERING)
# ---------------------------------------------------------------------------
separator("6. Slicing & Boolean Masking")

data = np.array([
    [10, 15, 20, 25],
    [30, 35, 40, 45],
    [50, 55, 60, 65]
])

# Multi-dimensional slicing: data[row_slice, col_slice]
# Get all rows, but only the first 2 feature columns:
X_features = data[:, :2]
print("All rows, first 2 columns:\n", X_features)

# Get the last column (e.g. target label y):
y_labels = data[:, -1]
print("\nTarget labels (last column):", y_labels)

# Boolean Masking (filtering without if-statements)
# E.g. find all numbers greater than 35:
mask = data > 35
print("\nBoolean Mask (data > 35):\n", mask)
print("Values where mask is True:", data[mask])

# ReLU activation function in 1 line using boolean masking or np.maximum:
# ReLU(x) = max(0, x)
raw_logits = np.array([-2.5, 0.0, 1.2, -0.4, 3.8])
relu_output = np.maximum(0, raw_logits)
print("\nReLU Activation applied to", raw_logits)
print("-->", relu_output)


# ---------------------------------------------------------------------------
# 7. AGGREGATIONS & AXES (SUM, MEAN, ARGMAX)
# ---------------------------------------------------------------------------
separator("7. Aggregations & Axes (axis=0 vs axis=1)")

scores = np.array([
    [85, 90, 78],
    [70, 88, 95],
    [92, 60, 80],
    [60, 75, 85]
])  # 4 students (rows), 3 exams (columns)

print("Scores Matrix (4 students x 3 exams):\n", scores)

# axis=0: Collapse down the rows -> Compute per-column (e.g., average exam scores)
print("\nMean of each exam (axis=0):", np.mean(scores, axis=0))

# axis=1: Collapse across the columns -> Compute per-row (e.g., student average)
print("Mean score per student (axis=1):", np.mean(scores, axis=1))

# argmax: index of maximum value (used for classification predictions!)
# e.g., which exam was each student's best?
best_exam_per_student = np.argmax(scores, axis=1)
print("Best exam index per student (np.argmax):", best_exam_per_student)


# ---------------------------------------------------------------------------
# 8. HANDS-ON CHALLENGE FOR YOU TO IMPLEMENT
# ---------------------------------------------------------------------------
separator("8. Mini Practice Exercises")
print("""
Try completing the two functions below, then run the script to test!
""")

def standardize(matrix: np.ndarray) -> np.ndarray:
    """
    Exercise 1: Standard Score (Z-Score Normalization)
    Formula: (X - mean) / std
    
    Standardize the columns (features) of a 2D matrix so each feature has:
    - Mean = 0
    - Standard deviation = 1
    
    Hint: Compute np.mean(..., axis=0) and np.std(..., axis=0),
          then rely on NumPy broadcasting to subtract and divide!
    """
    # TODO: Replace with your implementation
    return matrix


def predict_linear_layer(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Exercise 2: Linear Model Forward Pass
    Formula: Y = X @ W + b
    
    Given:
    - X: Input feature matrix of shape (N, D) where N is samples, D is features
    - W: Weight matrix of shape (D, M) where M is outputs
    - b: Bias vector of shape (M,)
    
    Return the computed predictions Y of shape (N, M).
    """
    # TODO: Replace with your implementation
    return X


if __name__ == "__main__":
    # Test Exercise 1: Standardize
    sample_data = np.array([
        [10.0, 200.0],
        [20.0, 400.0],
        [30.0, 600.0]
    ])
    print("Original data:\n", sample_data)
    
    std_data = standardize(sample_data)
    print("\nYour Standardized Data:\n", np.round(std_data, 2))
    print("Expected Standardized Data (approx):\n [[-1.22 -1.22]\n [ 0.    0.  ]\n [ 1.22  1.22]]")

    # Test Exercise 2: Linear Forward Pass
    test_X = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    test_W = np.array([[0.5], [1.5]])             # (2, 1)
    test_b = np.array([0.1])                     # (1,)
    
    pred = predict_linear_layer(test_X, test_W, test_b)
    print("\nYour Linear Prediction:\n", pred)
    print("Expected Linear Prediction:\n [[3.6]\n [7.6]]")
