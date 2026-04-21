# Neural Networks & LLM Fine-Tuning Assessment

---

## Section 1: Objective Questions

---

### Q1. Backpropagation

**Question:**
In neural network training, what is the primary role of the backpropagation algorithm?

**Options:**

1. To initialise all weights of the network to zero
2. To compute gradients of the loss with respect to each weight and propagate them backwards
3. To select the best architecture for the neural network
4. To convert input data into a format suitable for training

**Correct Answer:** Option 2

#### Explanation

Backpropagation computes the gradient of the loss function with respect to each weight using the **chain rule**, propagating errors from output to input. These gradients are then used to update weights via gradient descent.

#### Why Others Are Incorrect

* **Option 1:** Weight initialization is a separate step; zero initialization causes symmetry issues
* **Option 3:** Architecture selection is done before training
* **Option 4:** Data preprocessing is unrelated to backpropagation

---

### Q2. Learning Rate (η)

**Question:**
In the weight update equation
[
w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}
]
what does η represent?

**Options:**

1. Total number of training epochs
2. Gradient of the loss
3. Learning rate
4. Prediction error

**Correct Answer:** Option 3

#### Explanation

η (eta) is the **learning rate**, which controls the step size of weight updates.

* Large η → unstable training
* Small η → slow convergence

#### Why Others Are Incorrect

* **Option 1:** Epoch count is unrelated to per-step update
* **Option 2:** Gradient is ∂L/∂w
* **Option 4:** Error is part of loss, not η

---

### Q3. Fine-Tuning LLMs

**Question:**
Why is fine-tuning necessary for pre-trained LLMs?

**Options:**

1. Pre-trained LLMs are general and may underperform in specialised domains
2. LLMs cannot process text without fine-tuning
3. Fine-tuning adds new layers
4. LLMs are always overfitted

**Correct Answer:** Option 1

#### Explanation

Pre-trained models learn general patterns but lack domain-specific expertise. Fine-tuning adapts them to tasks like medical or legal applications.

#### Why Others Are Incorrect

* **Option 2:** LLMs can already process text
* **Option 3:** Fine-tuning updates weights, not architecture
* **Option 4:** They are not inherently overfitted

---

### Q4. LoRA Parameter Count

**Question:**
For LoRA with:

* D = 4096, K = 4096, r = 8

How many trainable parameters are introduced?

**Options:**

1. 16,777,216
2. 65,536
3. 32,768
4. 8

**Correct Answer:** Option 2

#### Explanation

* A: (4096 \times 8 = 32,768)
* B: (8 \times 4096 = 32,768)
* Total = **65,536**

#### Why Others Are Incorrect

* **Option 1:** Full matrix size
* **Option 3:** Only one matrix
* **Option 4:** Just rank value

---

### Q5. Q-LoRA Memory Reduction

**Question:**
What is the memory reduction when converting FP16 → 4-bit?

**Options:**

1. 2×
2. 8×
3. 4×
4. No reduction

**Correct Answer:** Option 3

#### Explanation

* FP16 = 16 bits
* 4-bit = 4 bits
  [
  16 / 4 = 4\times
  ]

#### Why Others Are Incorrect

* **Option 1:** 16→8 bit
* **Option 2:** would require 2-bit
* **Option 4:** incorrect

---

### Q6. Full Fine-Tuning Disadvantages (MCMC)

**Select all correct:**

1. Catastrophic forgetting
2. High compute cost
3. Overfitting risk
4. Cannot improve performance

**Correct Answer:** 1, 2, 3

#### Explanation

* Updating all weights → high cost
* Small data → overfitting
* Knowledge overwrite → forgetting

---

### Q7. LoRA Advantages (MCMC)

**Select all correct:**

1. Fewer parameters
2. Frozen base weights
3. Swappable adapters
4. Always better accuracy

**Correct Answer:** 1, 2, 3

---

### Q8. Q-LoRA Trade-offs (MCMC)

**Select all correct:**

1. Minor accuracy loss
2. Slightly slower training
3. More trainable params
4. Enables large model training

**Correct Answer:** 1, 2, 4

---

### Q9. LoRA Parameter Calculation (Integer)

**Question:**
Matrix: (2048 \times 2048), rank (r=4)

**Answer:**
[
(2048 \times 4) + (4 \times 2048) = 8192 + 8192 = 16,384
]

---

### Q10. Quantization Mapping (Integer)

**Question:**
Map weight 3.0 in range [-5, 5] using 4-bit quantization

[
\text{level} = \text{round} \left( \frac{w - w_{min}}{w_{max} - w_{min}} \times 15 \right)
]

**Answer:**
[
= \text{round}(8/10 \times 15) = \text{round}(12) = 12
]

---

## Section 2: Subjective Question

---

### Q11. Case Study — LLM Fine-Tuning Strategy

**Scenario:**
13B model, 15K samples, RTX 4090 (24GB)

---

### 1. Memory Estimates

| Method  | Estimate |
| ------- | -------- |
| Full FT | ~104 GB  |
| LoRA    | ~26 GB   |
| Q-LoRA  | ~6.6 GB  |

---

### 2. Feasibility

* Full FT → ❌ Not feasible
* LoRA → ⚠️ Borderline (needs optimizations)
* Q-LoRA → ✅ Feasible

---

### 3. Recommendation: **Q-LoRA**

* Memory efficient
* Preserves knowledge
* Lower overfitting risk

---

### 4. Quantization Process

* Convert FP16 → 4-bit levels (0–15)
* Store scale + integer
* Dequantize during forward pass

**Trade-off:** small approximation error

---

### 5. Multi-task Support

LoRA enables:

* Multiple adapters (medical, legal)
* Same base model reused
* Easy swapping at inference

