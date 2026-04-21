# Numeric Manipulation & Linear Algebra Basics
### Lecture Notes | AIM Program — IIT Patna × Masai School

---

## Session Overview

Data doesn't arrive in the shape you need. It comes as flat lists, nested lists, wide tables, tall tables — and before any model can touch it, you have to bend it into the right form. NumPy is the tool that makes this bending fast, expressive, and mathematically precise.

In this session, we move beyond basic array creation and indexing. We focus on the operations that data scientists perform dozens of times every week: reshaping arrays, transposing matrices, multiplying matrices, and generating synthetic data for experiments and testing. These aren't abstract exercises — they are the hidden plumbing behind every ML pipeline you'll ever build.

**Duration:** 2 hours  
**Tools Required:** Python 3.x, NumPy (`pip install numpy`), Google Colab or Jupyter Notebook

---

## Learning Objectives

By the end of this session, students will be able to:

1. Reshape and transpose NumPy arrays to match the input requirements of ML models and data pipelines.
2. Perform and interpret matrix multiplication using `np.dot()` and the `@` operator.
3. Generate synthetic datasets using NumPy's random module for experimentation and testing.
4. Apply these operations to realistic data transformation tasks in ML workflows.

---

## Concept Motivation: Why Shape Matters More Than You Think

Imagine you are a chef. You have all the right ingredients — fresh vegetables, spices, proteins. But if you try to cook a stir-fry using soup bowls instead of a wok, your technique will fail regardless of how good the ingredients are.

Arrays in machine learning work the same way. A neural network layer expects input of shape `(batch_size, features)`. A convolutional layer expects `(batch_size, height, width, channels)`. A matrix multiplication requires the inner dimensions to match. If your array is the wrong shape, nothing works — even if the numbers inside are correct.

**Real-world scenario:** You're working on a customer churn model. Your raw data is a flat list of 1,000 feature values from 10 customers, each with 100 features. Before you pass it to sklearn's `LogisticRegression`, you need it shaped as `(10, 100)`. Before you pass it to a PyTorch `Linear` layer, you might need it as `(10, 100)` or `(100, 10)` depending on the layer definition. One wrong reshape and your model silently learns garbage.

This session teaches you to think in shapes — and to transform data with confidence.

---

## Core Concept 1: Array Reshaping

### What Is Reshaping?

Reshaping means changing the dimensional structure of an array without changing its underlying data. The total number of elements must remain constant.

A 1D array of 12 elements can be reshaped into:
- `(2, 6)` — 2 rows, 6 columns
- `(3, 4)` — 3 rows, 4 columns
- `(4, 3)` — 4 rows, 3 columns
- `(2, 2, 3)` — 3D array
- `(12,)` — back to 1D

The rule: **product of new shape dimensions = product of old shape dimensions**.

### The `reshape()` Method

```python
import numpy as np

# 1D array of 12 elements
arr = np.arange(12)
print(arr)         # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(arr.shape)   # (12,)

# Reshape to 2D: 3 rows, 4 columns
arr_2d = arr.reshape(3, 4)
print(arr_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(arr_2d.shape)  # (3, 4)

# Reshape to 3D
arr_3d = arr.reshape(2, 2, 3)
print(arr_3d.shape)  # (2, 2, 3)
```

### The `-1` Wildcard

NumPy allows you to use `-1` in one dimension and it automatically calculates the correct size:

```python
arr = np.arange(24)

# Let NumPy figure out the number of rows
arr_2d = arr.reshape(-1, 6)   # Shape: (4, 6)

# Let NumPy figure out the number of columns
arr_2d = arr.reshape(4, -1)   # Shape: (4, 6)

# Flatten to 1D
arr_flat = arr_2d.reshape(-1)  # Shape: (24,)
```

This is especially useful when you don't know one dimension in advance — common when working with variable batch sizes.

### `flatten()` vs `ravel()`

Both produce a 1D array, but they behave differently in memory:

```python
arr = np.array([[1, 2], [3, 4]])

flat = arr.flatten()   # Returns a copy — changes don't affect original
rav  = arr.ravel()     # Returns a view — changes may affect original

flat[0] = 99
print(arr[0, 0])  # Still 1 — flatten made a copy

rav[0] = 99
print(arr[0, 0])  # 99 — ravel gave a view
```

**Rule of thumb:** Use `flatten()` when you need safety. Use `ravel()` when you need speed and memory efficiency.

### ML Application: Preparing Image Data

Images are stored as 3D arrays `(height, width, channels)`. Neural networks often need them flattened per sample:

```python
# Simulating 100 grayscale images, each 28x28
images = np.random.randint(0, 256, size=(100, 28, 28))
print(images.shape)  # (100, 28, 28)

# Flatten each image for a fully connected layer
images_flat = images.reshape(100, -1)
print(images_flat.shape)  # (100, 784)
```

This is exactly what happens inside MNIST preprocessing.

---

## Core Concept 2: Transposition

### What Is a Transpose?

Transposition flips a matrix along its diagonal — rows become columns and columns become rows.

For a matrix of shape `(m, n)`, its transpose has shape `(n, m)`.

```
Original (2×3):          Transposed (3×2):
[[1, 2, 3],              [[1, 4],
 [4, 5, 6]]               [2, 5],
                           [3, 6]]
```

### Using `.T` and `np.transpose()`

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)      # (2, 3)

AT = A.T
print(AT)
# [[1, 4]
#  [2, 5]
#  [3, 6]]
print(AT.shape)     # (3, 2)

# Equivalent: np.transpose()
AT2 = np.transpose(A)
print(np.array_equal(AT, AT2))  # True
```

### Transposing Higher-Dimensional Arrays

For 3D+ arrays, `.T` reverses all axes. To specify custom axis ordering, use `np.transpose(arr, axes=(…))`:

```python
arr = np.zeros((2, 3, 4))
print(arr.T.shape)                        # (4, 3, 2) — reverses all axes
print(np.transpose(arr, (0, 2, 1)).shape) # (2, 4, 3) — swap axes 1 and 2
```

### Mathematical Intuition

Transposition is fundamental to several ML operations:

- **Dot product:** `x · y = xᵀy`
- **Covariance matrix:** `Σ = XᵀX / n`
- **Gradient computation:** Backpropagation relies heavily on transposed weight matrices

### ML Application: Feature Matrix Alignment

```python
# X: 5 samples × 3 features
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15]])

# Covariance-style computation: XᵀX gives (3×3) feature correlation
cov_approx = X.T @ X
print(cov_approx.shape)  # (3, 3)
```

---

## Core Concept 3: Matrix Multiplication

### Why Matrix Multiplication?

Matrix multiplication is the engine of linear algebra — and by extension, the engine of neural networks. Every linear layer in a neural network computes:

```
output = input @ weights + bias
```

Understanding how shapes interact in matrix multiplication is non-negotiable for any ML practitioner.

### The Rule: Inner Dimensions Must Match

For two matrices A of shape `(m, k)` and B of shape `(k, n)`:
- The inner dimensions must match: `k == k`
- The result has shape `(m, n)`

```
A (m × k)  @  B (k × n)  =  C (m × n)
```

Think of it as: **(rows of A) × (columns of B)** — you slide each row of A across each column of B.

### Mathematical Definition

For matrices A and B, element `C[i, j]` of the product is:

```
C[i, j] = Σ A[i, k] * B[k, j]   for k = 0 to K-1
```

Each output element is a dot product between a row of A and a column of B.

### Implementation: Three Ways

```python
A = np.array([[1, 2],
              [3, 4]])  # Shape (2, 2)

B = np.array([[5, 6],
              [7, 8]])  # Shape (2, 2)

# Method 1: np.dot()
C1 = np.dot(A, B)

# Method 2: @ operator (preferred in modern Python)
C2 = A @ B

# Method 3: np.matmul()
C3 = np.matmul(A, B)

print(C1)
# [[19, 22]
#  [43, 50]]
```

**Prefer `@`** in new code — it is cleaner and more readable.

### Element-wise vs Matrix Multiplication

A common source of bugs: confusing `*` (element-wise) with `@` (matrix):

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A * B)   # Element-wise: [[5,12],[21,32]]
print(A @ B)   # Matrix mult:  [[19,22],[43,50]]
```

### Batch Matrix Multiplication

For 3D arrays (batches of matrices), `np.matmul()` / `@` handles batch dimensions automatically:

```python
# 10 samples, each with a 4×3 matrix
batch_A = np.random.randn(10, 4, 3)
# Weight matrix: 3×5
batch_B = np.random.randn(10, 3, 5)

result = batch_A @ batch_B
print(result.shape)  # (10, 4, 5) — multiplication applied to each sample
```

### ML Application: Simulating a Linear Layer

```python
# 8 input samples, 4 features each
X = np.random.randn(8, 4)

# Weight matrix: 4 inputs → 3 outputs
W = np.random.randn(4, 3)

# Bias: one per output neuron
b = np.random.randn(3)

# Forward pass through a linear layer
output = X @ W + b
print(output.shape)  # (8, 3) — 8 samples, 3 output values each
```

This is exactly what `torch.nn.Linear` does internally.

---

## Core Concept 4: Generating Synthetic Data

### Why Synthetic Data?

Before you have real data — or when you want to test an idea quickly — synthetic data is your best friend. NumPy's random module lets you generate controlled, reproducible datasets that mimic real distributions.

Key use cases:
- Unit testing ML pipelines
- Teaching and demonstrations
- Stress testing with large volumes
- Simulating specific distributions (Gaussian, uniform, Poisson)

### Setting the Random Seed

Always set a seed for reproducibility:

```python
np.random.seed(42)   # Traditional approach
rng = np.random.default_rng(42)  # Modern approach (preferred)
```

With the same seed, you get the same numbers every time — critical for debugging and sharing experiments.

### Common Distributions

```python
rng = np.random.default_rng(42)

# Uniform distribution [0, 1)
uniform = rng.random((3, 4))

# Normal (Gaussian) distribution: mean=0, std=1
normal = rng.standard_normal((3, 4))

# Normal with custom mean and std
custom_normal = rng.normal(loc=5.0, scale=2.0, size=(3, 4))

# Integers in [low, high)
integers = rng.integers(low=0, high=100, size=(3, 4))

# Binomial distribution (n trials, p probability)
binomial = rng.binomial(n=10, p=0.3, size=(3, 4))
```

### Generating a Synthetic Classification Dataset

```python
np.random.seed(0)
n_samples = 200

# Class 0: centered at (2, 2)
class_0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])

# Class 1: centered at (-2, -2)
class_1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])

# Stack features and create labels
X = np.vstack([class_0, class_1])         # Shape: (200, 2)
y = np.hstack([np.zeros(100), np.ones(100)])  # Shape: (200,)

print(X.shape, y.shape)  # (200, 2) (200,)
```

### Generating a Synthetic Regression Dataset

```python
np.random.seed(1)
n = 100

# True relationship: y = 3x + noise
X = np.linspace(0, 10, n).reshape(-1, 1)   # Shape: (100, 1)
noise = np.random.randn(n) * 1.5
y = 3 * X.squeeze() + 7 + noise            # Shape: (100,)
```

### Useful Utility Functions

```python
# Linearly spaced values
x = np.linspace(0, 1, 50)       # 50 points from 0 to 1

# Range with step
x = np.arange(0, 10, 0.5)       # 0, 0.5, 1.0, ..., 9.5

# Zeros, ones, identity matrix
Z = np.zeros((3, 4))
O = np.ones((3, 4))
I = np.eye(4)                    # 4×4 identity matrix

# Diagonal matrix from a vector
D = np.diag([1, 2, 3, 4])       # 4×4 diagonal matrix
```

---

## Visual Explanation: Putting It All Together

Consider a simplified ML preprocessing pipeline:

```
Raw Data           Reshaped            Transposed           Multiplied
(1000,)    →    (100, 10)      →     (10, 100)       →   (10, 100) @ (100, 5)
Flat list    Feature matrix     Columns become rows    Linear projection to 5D
```

Each step is one NumPy operation. Together, they define data transformation.

---

## Real-World Applications

| Operation | Where It's Used |
|---|---|
| `reshape()` | Flattening images for dense layers; batch processing |
| `.T` / `transpose()` | Covariance computation; attention score alignment |
| `@` / `matmul()` | Linear layers; PCA; attention mechanisms |
| `random.randn()` | Weight initialization; synthetic benchmarks |
| `linspace()` | Creating test inputs; plotting curves |
| `np.eye()` | Identity operations; regularization terms |

---

## Implementation Overview

Below is a complete mini-pipeline combining all four concepts:

```python
import numpy as np

np.random.seed(42)

# Step 1: Generate synthetic data — 50 samples, 6 features
X_flat = np.random.randn(300)           # 300 values flat
X = X_flat.reshape(50, 6)              # Reshape to (50, 6)
print("Feature matrix:", X.shape)      # (50, 6)

# Step 2: Create a weight matrix for a linear projection
W = np.random.randn(6, 4)             # 6 inputs → 4 outputs
print("Weight matrix:", W.shape)      # (6, 4)

# Step 3: Linear projection (simulating a layer)
Z = X @ W                             # (50, 6) @ (6, 4) = (50, 4)
print("Projected output:", Z.shape)   # (50, 4)

# Step 4: Transpose for downstream computation
Z_T = Z.T                             # (4, 50)
print("Transposed output:", Z_T.shape)

# Step 5: Compute a covariance-like matrix
cov = Z_T @ Z                         # (4, 50) @ (50, 4) = (4, 4)
print("Covariance matrix:", cov.shape)
```

---

## Key Takeaways

1. **Reshaping** changes the shape of an array without changing its data. Always verify the total element count is preserved. Use `-1` as a wildcard for one unknown dimension.

2. **Transposition** swaps rows and columns. It is essential for aligning shapes in matrix operations and computing covariance matrices.

3. **Matrix multiplication** (`@`) requires inner dimensions to match: `(m, k) @ (k, n) → (m, n)`. It is fundamentally different from element-wise multiplication (`*`).

4. **Synthetic data** enables reproducible experiments. Always set a random seed. Choose the distribution that matches your use case — normal for features, uniform for ranges, integer for categories.

5. **Shape awareness** is a professional skill. Before every operation, know your input shapes and expected output shapes. When in doubt, print `.shape`.

---

## Practice Prompts (Reflective)

- A dataset has shape `(500, 1, 28, 28)`. What does each dimension represent? How would you reshape it for a dense layer?
- Why does `(3, 4) @ (4, 5)` work but `(3, 4) @ (3, 5)` fail?
- When would you prefer `ravel()` over `flatten()` in a production pipeline?
- If your weight matrix has shape `(10, 5)`, what input shape does it expect? What output shape does it produce?

---

*Session prepared for the AIM Program — IIT Patna × Masai School*
