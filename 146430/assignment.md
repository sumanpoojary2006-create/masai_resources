

# NumPy Assessment

---

## Section 1: Objective Questions

---

### Q1. `np.sum(arr)`

**Question:**
What is the output of the following code?

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))
```

**Options:**

1. 5
2. 15
3. 120
4. 25

**Correct Answer:** 15

#### Explanation

`np.sum()` computes the total of all elements in the array.

[
1 + 2 + 3 + 4 + 5 = 15
]

#### Why Correct

* Adds all elements → returns **15**

#### Why Others Are Wrong

* **5** → number of elements
* **120** → product (`np.prod`)
* **25** → sum of squares

---

### Q2. Boolean Indexing

**Question:**
Given:

```python
arr = np.array([10, 20, 30, 40, 50])
```

What does `arr[arr > 25]` return?

**Options:**

1. [10, 20]
2. [10, 20, 30]
3. [30, 40, 50]
4. [True, True, False, False, False]

**Correct Answer:** `[30, 40, 50]`

#### Explanation

```python
arr > 25 → [False, False, True, True, True]
arr[arr > 25] → [30, 40, 50]
```

#### Why Others Are Wrong

* `[10, 20]` → incorrect condition
* `[10, 20, 30]` → partially filtered
* Boolean array → mask, not result

---

### Q3. Reshape

**Question:**

```python
arr = np.array([1, 10, 100, 1000, 10000, 100000])
result = arr.reshape((2, 3))
```

What is the shape?

**Options:**

1. (2, 3)
2. (3, 2)
3. (6,)
4. (1, 6)

**Correct Answer:** (2, 3)

#### Explanation

[
2 \times 3 = 6 \text{ elements}
]

#### Why Others Are Wrong

* `(3,2)` → different reshape
* `(6,)` → original
* `(1,6)` → incorrect shape

---

### Q4. Dot Product

**Question:**

```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print(np.dot(A, B))
```

**Options:**

1. 15
2. [4, 10, 18]
3. 32
4. 12

**Correct Answer:** 32

#### Explanation

[
(1×4) + (2×5) + (3×6) = 32
]

#### Why Others Are Wrong

* 15 → sum of A
* [4,10,18] → element-wise
* 12 → incorrect

---

### Q5. `np.argmax(arr)`

**Question:**

```python
arr = np.array([15, 42, 7, 99, 23])
```

What does `np.argmax(arr)` return?

**Options:**

1. 99
2. 0
3. 2
4. 3

**Correct Answer:** 3

#### Explanation

* Max = 99
* Index = 3

---

### Q6. Concatenation

**Question:**

```python
result = np.concatenate((A, B), axis=1)
```

**Options:**

1. (4, 2)
2. (2, 4)
3. (4, 4)
4. (2, 2)

**Correct Answer:** (2, 4)

#### Explanation

[
(2,2) + (2,2) \rightarrow (2,4)
]

---

### Q7. Invalid Reshape

**Question:**
Which reshape raises an error?

**Options:**

1. (2,3)
2. (1,6)
3. (4,2)
4. (3,2)

**Correct Answer:** (4,2)

#### Explanation

* Total elements must match
* 4×2 = 8 ≠ 6 → ❌

---

### Q8. `np.split()` (MCMC)

**Correct Answers:** 1, 2, 3

#### Explanation

* Equal split required
* axis=0 → row-wise
* 6 elements → 3 splits → 2 each

---

### Q9. Element-wise Operations (MCMC)

**Correct Answers:** 1, 3, 4

#### Explanation

* `np.sqrt` → element-wise
* `np.exp` → element-wise
* `np.cumsum` → cumulative

❌ `np.sum` → reduction

---

### Q10. Flatten

**Question:**
Shape (3,4) → flatten size?

**Answer:** 12

#### Explanation

[
3 \times 4 = 12
]

---

## Section 2: Coding Question

---

### Q11. NumPy Operations Script

#### Problem Statement

Write a Python script that:

1. Creates a random array (size 12, seed 42)
2. Reshapes to (3,4)
3. Filters values > 50
4. Concatenates along axis=0
5. Splits into 4 parts
6. Computes statistics
7. Computes dot product with another array (seed 7)

---

### Reference Solution

```python
import numpy as np

# Step 1
np.random.seed(42)
arr = np.random.randint(1, 101, 12)
print('1. Original array:', arr)

# Step 2
arr_2d = arr.reshape((3, 4))
print('2. Reshaped:\n', arr_2d)

# Step 3
print('3. >50:', arr[arr > 50])

# Step 4
concat = np.concatenate((arr_2d, arr_2d), axis=0)
print('4. Shape:', concat.shape)

# Step 5
parts = np.split(arr, 4)
for i, p in enumerate(parts):
    print(f'Part {i+1}:', p)

# Step 6
print('Sum:', np.sum(arr))
print('Mean:', np.mean(arr))
print('Max:', np.max(arr))
print('Min:', np.min(arr))
print('Argmax:', np.argmax(arr))
print('Argmin:', np.argmin(arr))

# Step 7
np.random.seed(7)
arr2 = np.random.randint(1, 101, 12)
print('Dot product:', np.dot(arr, arr2))
```

---

## Submission Guidelines

* Use **Google Colab**
* Share link with **view access**
* Include:

  * Problem statement
  * Clean code
  * Comments
  * Executed outputs

