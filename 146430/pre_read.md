# Pre-Read: Numeric Manipulation & Linear Algebra Basics

**Duration:** 30-40 minutes  
**Topic:** Array reshaping, transposition, matrix operations, and data generation  
**Prerequisites:** NumPy 2D arrays, basic indexing

---

## Introduction: Why These Operations Matter

You've learned to create and manipulate arrays. Now you'll learn to **transform** them - changing their shape, orientation, and combining them in powerful ways.

**Real-world context:**

Imagine you're a data scientist at Netflix. You have:
- User viewing history: 1 million users × 10,000 movies
- Need to reshape data for machine learning algorithms
- Must transpose matrices for mathematical operations
- Want to generate synthetic test data before deploying to production

These operations are the **foundation of data science and machine learning**. Every neural network, recommendation system, and image processor uses them constantly.

---

## 1. Array Reshaping: Changing Dimensions Without Changing Data

### What is Reshaping?

**Reshaping** means changing the dimensions (shape) of an array **without changing the data itself**.

Think of it like rearranging books on shelves:
- You have 12 books
- Currently: 2 shelves × 6 books each (shape: 2, 6)
- Reshape to: 3 shelves × 4 books each (shape: 3, 4)
- **Same 12 books, different arrangement**

### Why Reshape?

**Real-world scenario: Image Processing**

```python
# Camera captures image as 1D stream of pixels
raw_data = np.array([255, 200, 180, ..., 100])  # 921,600 values

# Need to reshape to actual image dimensions
# HD image: 1280 pixels wide × 720 pixels tall
image = raw_data.reshape(720, 1280)

# Now can display as 2D image!
```

**Machine Learning scenario:**

```python
# You have 1000 samples, each with 28×28 pixel images
# Current shape: (1000, 28, 28) - 3D array
data = load_images()

# Neural network needs 1D input: flatten each image
# Target shape: (1000, 784) where 784 = 28 × 28
data_flat = data.reshape(1000, 784)

# Now ready for machine learning!
```

### Basic Reshaping Syntax

```python
import numpy as np

# Create 1D array with 12 elements
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(arr.shape)  # (12,)

# Reshape to 2D: 3 rows × 4 columns
arr_2d = arr.reshape(3, 4)
print(arr_2d)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
print(arr_2d.shape)  # (3, 4)

# Reshape to 2D: 4 rows × 3 columns
arr_2d_alt = arr.reshape(4, 3)
print(arr_2d_alt)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
print(arr_2d_alt.shape)  # (4, 3)
```

**Key insight:** The total number of elements must stay the same!
- 12 elements can reshape to (3, 4), (4, 3), (2, 6), (6, 2), (12, 1), etc.
- Cannot reshape to (3, 5) - that would need 15 elements!

### The Magic of -1: Automatic Size Calculation

NumPy can calculate one dimension for you using `-1`:

```python
arr = np.arange(24)  # 24 elements

# Let NumPy figure out the columns
arr_2d = arr.reshape(4, -1)
print(arr_2d.shape)  # (4, 6)
# NumPy calculated: 24 ÷ 4 = 6 columns

# Let NumPy figure out the rows
arr_2d = arr.reshape(-1, 8)
print(arr_2d.shape)  # (3, 8)
# NumPy calculated: 24 ÷ 8 = 3 rows

# Flatten to 1D (most common use of -1)
arr_flat = arr_2d.reshape(-1)
print(arr_flat.shape)  # (24,)
```

**When is this useful?**

```python
# Reading batch of images from disk
images = load_batch()
print(images.shape)  # (?, 28, 28) - unknown number of images

# Flatten each image, let NumPy figure out how many
num_pixels = 28 * 28
flat_images = images.reshape(-1, num_pixels)
# NumPy automatically calculates first dimension
print(flat_images.shape)  # (100, 784) if 100 images loaded
```

### Reshaping Rules

**Rule 1: Total elements must match**
```python
arr = np.arange(12)  # 12 elements

# ✓ Valid reshapes:
arr.reshape(3, 4)    # 3 × 4 = 12 ✓
arr.reshape(2, 6)    # 2 × 6 = 12 ✓
arr.reshape(12, 1)   # 12 × 1 = 12 ✓

# ✗ Invalid reshape:
arr.reshape(3, 5)    # 3 × 5 = 15 ✗ (need 12!)
# ValueError: cannot reshape array
```

**Rule 2: Data order is preserved (row-major)**

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape fills rows first (left to right, top to bottom)
arr.reshape(2, 3)
# [[1, 2, 3],    ← First row: elements 0-2
#  [4, 5, 6]]    ← Second row: elements 3-5

arr.reshape(3, 2)
# [[1, 2],       ← First row: elements 0-1
#  [3, 4],       ← Second row: elements 2-3
#  [5, 6]]       ← Third row: elements 4-5
```

### Common Reshaping Patterns

**1. Flattening (multi-D → 1D)**

```python
# 2D image to 1D vector
image = np.random.rand(28, 28)  # 28×28 image
vector = image.reshape(-1)      # Now 784 element vector

# Same as:
vector = image.flatten()
vector = image.ravel()  # Fastest method
```

**Use case:** Preparing image data for traditional machine learning

**2. Adding a dimension**

```python
# Single sample to batch of 1
sample = np.random.rand(28, 28)     # Shape: (28, 28)
batch = sample.reshape(1, 28, 28)   # Shape: (1, 28, 28)

# Or using newaxis (more readable)
batch = sample[np.newaxis, :, :]
```

**Use case:** Deep learning models expect batches, even for single predictions

**3. Batching data**

```python
# 1000 samples, reshape into batches of 100
data = np.random.rand(1000, 64)  # 1000 samples, 64 features each

# Reshape to 10 batches of 100 samples
batches = data.reshape(10, 100, 64)
# Now can process batch-by-batch
```

---

## 2. Transposition: Flipping Rows and Columns

### What is Transposition?

**Transposition** flips an array along its diagonal - rows become columns, columns become rows.

**Visual example:**

```
Original array:           Transposed array:
[[1, 2, 3],              [[1, 4],
 [4, 5, 6]]       →       [2, 5],
                          [3, 6]]

Shape: (2, 3)            Shape: (3, 2)
2 rows, 3 cols           3 rows, 2 cols
```

**Think of it as rotating a table:**
- Original: Students (rows) × Subjects (columns)
- Transposed: Subjects (rows) × Students (columns)

### Why Transpose?

**Real-world scenario 1: Matrix multiplication requirements**

Many mathematical operations require specific orientations:

```python
# Feature matrix: 100 samples × 5 features
X = np.random.rand(100, 5)

# Weight vector: 5 weights
w = np.random.rand(5)

# Matrix multiplication X @ w won't work as-is
# Need to transpose to align dimensions
```

**Real-world scenario 2: Data reorganization**

```python
# Data collected as: Time (rows) × Sensors (columns)
# 24 hours × 10 sensors
sensor_data = np.random.rand(24, 10)

# Need: Sensors (rows) × Time (columns) for analysis
sensor_data_T = sensor_data.T

# Now can analyze each sensor's time series easily
sensor_1_timeline = sensor_data_T[0, :]  # All times for sensor 1
```

### Transposition Syntax

```python
# Method 1: .T attribute (most common)
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
arr_T = arr.T

# Method 2: .transpose() method
arr_T = arr.transpose()

# Method 3: np.transpose() function
arr_T = np.transpose(arr)

# All produce the same result:
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### Transposition Properties

**Property 1: Shape flips**

```python
arr.shape = (3, 4)
arr.T.shape = (4, 3)  # Flipped!

arr.shape = (5, 2)
arr.T.shape = (2, 5)  # Flipped!
```

**Property 2: Double transpose returns original**

```python
arr = np.array([[1, 2], [3, 4]])

arr_T = arr.T          # Transpose
arr_T_T = arr_T.T      # Transpose again

print(np.array_equal(arr, arr_T_T))  # True
# Transposing twice gives you back the original
```

**Property 3: Transpose is a view (not a copy)**

```python
arr = np.array([[1, 2], [3, 4]])
arr_T = arr.T

# Modifying transpose modifies original!
arr_T[0, 0] = 999

print(arr)
# [[999, 2],
#  [  3, 4]]
# Original changed too!
```

### When to Transpose

**Use Case 1: Mathematical correctness**

```python
# Linear regression: y = Xw
# X shape: (100, 5) - 100 samples, 5 features
# w shape: (5,) - 5 weights

# Calculate predictions
predictions = X @ w  # Matrix multiplication works!

# But if X was transposed by mistake...
X_wrong = X.T  # Shape: (5, 100)
# predictions = X_wrong @ w  # Error! Shapes don't align

# Fix: Transpose back
X_correct = X_wrong.T
predictions = X_correct @ w  # Works!
```

**Use Case 2: Data format conversion**

```python
# Data from database: Features as rows (5 features × 100 samples)
data_db = np.random.rand(5, 100)

# Machine learning libraries expect: Samples as rows
data_ml = data_db.T  # Now (100, 5)

# Can now use with sklearn, tensorflow, etc.
```

**Use Case 3: Efficient column/row access**

```python
# Time series: 1000 timestamps × 50 sensors
time_series = np.random.rand(1000, 50)

# Want to analyze each sensor's full timeline
# Accessing columns is slower than rows in NumPy

# Transpose so sensors are rows
ts_transposed = time_series.T  # (50, 1000)

# Now accessing each sensor's data is faster
sensor_1 = ts_transposed[0]  # Fast row access
```

---

## 3. Matrix Multiplication: Combining Arrays

### What is Matrix Multiplication?

Matrix multiplication is **NOT** element-wise multiplication. It's a mathematical operation that combines arrays in a specific way.

**Visual example:**

```
A = [[1, 2],       B = [[5, 6],
     [3, 4]]            [7, 8]]

A × B = [[1×5 + 2×7,  1×6 + 2×8],
         [3×5 + 4×7,  3×6 + 4×8]]

      = [[19, 22],
         [43, 50]]
```

### Element-wise vs Matrix Multiplication

**Element-wise multiplication (A * B):**
- Multiply corresponding elements
- Arrays must have **same shape**
- Symbol: `*`

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A * B  # Element-wise
# [[1×5, 2×6],
#  [3×7, 4×8]]
# = [[5, 12],
#    [21, 32]]
```

**Matrix multiplication (A @ B):**
- Dot product of rows and columns
- A's columns must match B's rows
- Symbol: `@`

```python
A = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)
B = np.array([[5, 6], [7, 8]])  # Shape: (2, 2)

C = A @ B  # Matrix multiplication
# [[1×5 + 2×7,  1×6 + 2×8],
#  [3×5 + 4×7,  3×6 + 4×8]]
# = [[19, 22],
#    [43, 50]]
```

### Matrix Multiplication Rules

**Rule 1: Shape compatibility**

For `A @ B` to work:
- A's **number of columns** must equal B's **number of rows**

```python
# Compatible shapes:
A.shape = (3, 4)  # 4 columns
B.shape = (4, 5)  # 4 rows ✓
C = A @ B
C.shape = (3, 5)  # Result: outer dimensions

# Incompatible shapes:
A.shape = (3, 4)  # 4 columns
B.shape = (5, 2)  # 5 rows ✗
# A @ B → Error!
```

**Rule 2: Result shape**

```
(m, n) @ (n, p) = (m, p)

A: (3, 4) @ B: (4, 5) = C: (3, 5)
    ↑           ↑           ↑  ↑
    |           |           |  |
    m           n           m  p
                |
                Must match!
```

**Visual:**
```
A (3 rows, 4 cols)  @  B (4 rows, 5 cols)  =  C (3 rows, 5 cols)

    [4 cols]              [5 cols]              [5 cols]
    --------              --------              --------
    [     ]      @        [     ]      =        [     ]   3 rows
    [     ]               [     ]               [     ]
    [     ]               [     ]               [     ]
                          [     ]
    
    These must match! →   4 rows
```

### Why Matrix Multiplication Matters

**Real-world application: Recommendation systems**

```python
# User preferences: 1000 users × 50 features
users = np.random.rand(1000, 50)

# Movie features: 50 features × 10000 movies
movies = np.random.rand(50, 10000)

# Calculate all user-movie scores
# Shape: (1000, 50) @ (50, 10000) = (1000, 10000)
scores = users @ movies

# Result: Each user's predicted rating for each movie
# scores[0, 5] = User 0's predicted rating for Movie 5
```

**Real-world application: Neural networks**

```python
# Input layer: 100 samples × 784 features (28×28 images)
X = np.random.rand(100, 784)

# First layer weights: 784 inputs × 128 neurons
W1 = np.random.rand(784, 128)

# Compute first layer activations
# (100, 784) @ (784, 128) = (100, 128)
layer1 = X @ W1

# Each of 100 samples now has 128 neuron activations
```

### Common Operations

**Vector-matrix multiplication:**

```python
# Single sample (vector)
x = np.random.rand(784)  # Shape: (784,)

# Weight matrix
W = np.random.rand(784, 128)  # Shape: (784, 128)

# Prediction
output = x @ W  # Shape: (128,)
# Single output vector with 128 values
```

**Batch-matrix multiplication:**

```python
# Batch of samples
X = np.random.rand(100, 784)  # 100 samples

# Same weight matrix
W = np.random.rand(784, 128)

# Batch prediction
outputs = X @ W  # Shape: (100, 128)
# 100 output vectors, each with 128 values
```

---

## 4. Generating Synthetic Data

### Why Generate Synthetic Data?

**Before deploying to production:**
- Test algorithms without real data
- Create training data when real data is scarce
- Simulate scenarios (best case, worst case, edge cases)
- Benchmark performance with known characteristics

### Random Number Generation

**1. Random floats (0 to 1):**

```python
# Single random number
x = np.random.rand()  # e.g., 0.7345

# 1D array of random numbers
arr = np.random.rand(5)
# [0.123, 0.456, 0.789, 0.234, 0.567]

# 2D array
matrix = np.random.rand(3, 4)
# [[0.12, 0.45, 0.78, 0.23],
#  [0.56, 0.89, 0.12, 0.45],
#  [0.78, 0.23, 0.56, 0.89]]
```

**2. Random integers:**

```python
# Random integers from 0 to 9
dice_rolls = np.random.randint(0, 10, size=100)

# Random integers from 60 to 100 (test scores)
scores = np.random.randint(60, 100, size=(30, 5))
# 30 students × 5 subjects
```

**3. Random from normal distribution:**

```python
# Mean=0, std=1 (standard normal)
data = np.random.randn(1000)

# Mean=100, std=15 (IQ scores)
iq_scores = np.random.normal(100, 15, size=1000)
```

### Generating Realistic Datasets

**Example 1: Customer data**

```python
# 1000 customers
n_customers = 1000

# Age: Normal distribution, mean=35, std=12
ages = np.random.normal(35, 12, n_customers).astype(int)
ages = np.clip(ages, 18, 80)  # Clip to reasonable range

# Income: Log-normal (skewed distribution)
incomes = np.random.lognormal(10.5, 0.5, n_customers)

# Purchase amount: Uniform 10-500
purchases = np.random.uniform(10, 500, n_customers)

# Combine into dataset
customer_data = np.column_stack([ages, incomes, purchases])
# Shape: (1000, 3)
```

**Example 2: Time series data**

```python
# 365 days of stock prices
days = 365
start_price = 100

# Random daily returns (mean=0.1%, std=2%)
returns = np.random.normal(0.001, 0.02, days)

# Calculate cumulative price
prices = start_price * np.cumprod(1 + returns)
```

**Example 3: Image-like data**

```python
# Generate synthetic 28×28 grayscale images
n_images = 1000
image_data = np.random.rand(n_images, 28, 28)

# Add some structure (simple pattern)
for i in range(n_images):
    # Random horizontal line
    row = np.random.randint(0, 28)
    image_data[i, row, :] = 1.0
```

### Setting Random Seed (Reproducibility)

```python
# Without seed: different results each time
arr1 = np.random.rand(5)
arr2 = np.random.rand(5)
# arr1 ≠ arr2

# With seed: same results every time
np.random.seed(42)
arr1 = np.random.rand(5)

np.random.seed(42)
arr2 = np.random.rand(5)
# arr1 == arr2 (exactly the same!)

# Use in practice:
np.random.seed(42)  # Set once at start of script
# All subsequent random operations are reproducible
```

**Why this matters:**
- Debugging: Same random data every run
- Testing: Consistent test cases
- Research: Reproducible experiments
- Collaboration: Share code that produces same results

---

## 5. Putting It All Together

### Complete Example: Preparing ML Dataset

```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic customer data
n_samples = 1000

# Features: age, income, purchase_history
age = np.random.normal(35, 12, n_samples).astype(int)
age = np.clip(age, 18, 80)
income = np.random.lognormal(10.5, 0.5, n_samples)
purchase_history = np.random.randint(0, 50, n_samples)

# Combine features (samples × features)
X = np.column_stack([age, income, purchase_history])
print(f"Features shape: {X.shape}")  # (1000, 3)

# Generate target variable (will they buy premium?)
# Higher age + income → more likely to buy
prob = (age / 100 + income / 100000) / 2
y = (np.random.rand(n_samples) < prob).astype(int)
print(f"Target shape: {y.shape}")  # (1000,)

# Reshape target for matrix operations
y_col = y.reshape(-1, 1)
print(f"Target reshaped: {y_col.shape}")  # (1000, 1)

# Simulate linear model weights
W = np.random.randn(3, 1)  # 3 features → 1 output

# Make predictions
predictions = X @ W
print(f"Predictions shape: {predictions.shape}")  # (1000, 1)

# Transpose for different analysis
X_T = X.T
print(f"Transposed features: {X_T.shape}")  # (3, 1000)
# Now features are rows, samples are columns
```

---

## Key Concepts Summary

### Reshaping
- **Purpose:** Change array dimensions without changing data
- **Syntax:** `arr.reshape(new_shape)` or `arr.reshape(-1)` for auto-calc
- **Rule:** Total elements must stay same
- **Common:** Flattening (multi-D → 1D), batching, adding dimensions

### Transposition
- **Purpose:** Flip rows and columns
- **Syntax:** `arr.T` or `arr.transpose()`
- **Effect:** Shape (m, n) → (n, m)
- **Common:** Preparing data for matrix operations, reformatting

### Matrix Multiplication
- **Symbol:** `@` (NOT `*`)
- **Rule:** (m, n) @ (n, p) = (m, p)
- **Uses:** Neural networks, linear models, recommendations
- **Different from:** Element-wise multiplication (`*`)

### Synthetic Data
- **Functions:** `rand()`, `randint()`, `randn()`, `normal()`
- **Purpose:** Testing, simulation, augmentation
- **Best practice:** Use `np.random.seed()` for reproducibility
- **Realistic:** Combine multiple distributions for real-world patterns

---

## Pre-Read Quiz (Self-Check)

Test your understanding before the lecture:

**Question 1:** If an array has shape (24,), which reshapes are valid?
- A) (4, 6)
- B) (3, 8)
- C) (5, 5)
- D) (2, 12)

**Question 2:** What is the shape of `arr.T` if `arr.shape = (5, 3)`?

**Question 3:** Can you multiply a (100, 5) matrix with a (3, 10) matrix using `@`? Why or why not?

**Question 4:** What does `np.random.seed(42)` do?

**Question 5:** What's the difference between `*` and `@` for arrays?

### Answers:

**1:** A, B, D are valid (4×6=24, 3×8=24, 2×12=24). C is invalid (5×5=25≠24)

**2:** (3, 5) - rows and columns flip

**3:** No - first matrix has 5 columns but second has 3 rows. They don't match.

**4:** Sets random seed for reproducible random number generation

**5:** `*` is element-wise multiplication (same shape needed), `@` is matrix multiplication (inner dimensions must match)

---

## What's Next?

In the lecture, you'll:
1. Practice complex reshaping scenarios
2. Use transposition to solve real data problems
3. Build neural network layer computations with matrix multiplication
4. Generate complete ML-ready synthetic datasets
5. Combine all operations in production workflows

**Come prepared with questions about:**
- When to use which reshaping strategy
- How to debug shape mismatches in matrix multiplication
- Best practices for generating realistic synthetic data

---

**Remember:** These operations are the foundation of:
- Machine learning (reshaping data, matrix operations in neural nets)
- Data preprocessing (transposing for format conversion)
- Scientific computing (matrix algebra)
- Computer vision (image transformations)
- Recommendation systems (collaborative filtering)

Master these, and you'll understand 80% of what happens inside ML frameworks like TensorFlow and PyTorch!
