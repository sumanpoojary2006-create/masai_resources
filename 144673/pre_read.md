# Pre-Read: Fine-Tuning LLMs & AutoML

## The Startup That Couldn't Afford GPT-4

**Monday, 9:00 AM - TechStart AI Office, Mumbai**

Maya stared at her laptop screen, frustrated. Her startup had just landed a major client: a customer support platform serving 50,000 users. They needed an AI chatbot that could answer product-specific FAQs accurately.

"We can't just use GPT-4 API," her CTO Raj said, walking over. "The costs would eat our entire budget. We need to fine-tune our own model."

Maya groaned. "But fine-tuning a 7-billion parameter model... that needs what, 80GB of GPU memory? We only have a single A100 with 40GB."

Raj smiled. "Have you heard of LoRA?"

---

## The Memory Problem

Raj pulled up a diagram:

```
Traditional Fine-Tuning:
Model: 7B parameters
Full precision (FP32): 7B × 4 bytes = 28GB
Optimizer states: 28GB × 2 = 56GB
Gradients: 28GB
Total: ~112GB GPU memory

Our hardware: 40GB
Result: CRASH!
```

"That's the problem with full fine-tuning," Raj explained. "You need to update ALL parameters, store optimizer states, gradients... it's memory-intensive."

"So what's the solution?" Maya asked.

"We freeze 99.9% of the model. Only train a tiny fraction."

---

## Enter LoRA: The Clever Trick

**LoRA = Low-Rank Adaptation**

Raj drew on the whiteboard:

```
Standard fine-tuning:
W (Original weights) → W' (Updated weights)
Updates all 7 billion parameters!

LoRA:
W (Frozen!) + ΔW (Trainable)
ΔW = A × B

Where:
A: (d × r) - Down-projection matrix
B: (r × d) - Up-projection matrix  
r: Rank (typically 8, 16, 32, 64)

If d = 4096 and r = 16:
ΔW parameters = 4096 × 16 + 16 × 4096 = 131,072
Instead of 4096 × 4096 = 16,777,216

That's 128× fewer parameters!
```

"Wait," Maya interrupted. "You're saying instead of updating a 4096×4096 matrix, we create two small matrices that multiply together?"

"Exactly! And here's the magic: mathematically, you can approximate ANY weight update using low-rank decomposition. For most fine-tuning tasks, a rank of 16 is enough."

**The Results:**

```
Memory Usage:
Full fine-tuning: 112GB
LoRA fine-tuning: 14GB

Training Speed:
Full fine-tuning: 8 hours
LoRA fine-tuning: 2 hours

Model Quality:
Full fine-tuning: 94.2% accuracy
LoRA fine-tuning: 93.8% accuracy

Cost difference: 10× cheaper!
```

"We lose 0.4% accuracy but save 90% memory," Raj said. "Worth it?"

Maya nodded. "Definitely. But can we go even lower?"

---

## QLoRA: Going Even Smaller

"That's where QLoRA comes in," Raj said, pulling up another diagram.

**QLoRA = Quantized LoRA**

```
The Quantization Trick:

Normal model:
- Weights stored in FP32 (32 bits per number)
- 7B parameters × 4 bytes = 28GB

4-bit quantization:
- Weights stored in 4 bits
- 7B parameters × 0.5 bytes = 3.5GB

Memory savings: 8× reduction!
```

"Wait," Maya said. "Won't 4-bit precision destroy the model quality?"

Raj grinned. "That's the genius of QLoRA. They use **NF4 (NormalFloat4)** - a special 4-bit format optimized for neural network weights. Plus double quantization to save even more."

**QLoRA Memory Breakdown:**

```
Base model (4-bit quantized): 3.5GB
LoRA adapters (FP16): 0.3GB
Optimizer states: 1.2GB
Gradients: 0.8GB
Activation memory: 4GB
Total: ~10GB

Fits on a consumer GPU!
```

"So QLoRA = LoRA + Quantization," Maya summarized. "We get the memory savings of both techniques combined."

---

## The Hyperparameter Nightmare

Two weeks later, Maya had a new problem.

"Raj, I've run 50 training experiments. Different learning rates, LoRA ranks, batch sizes... I still can't beat 91% accuracy. There are too many hyperparameters!"

Raj pulled up a spreadsheet:

```
Hyperparameters to tune:
1. Learning rate: [1e-5, 1e-4, 1e-3]
2. LoRA rank (r): [4, 8, 16, 32, 64]
3. LoRA alpha: [8, 16, 32, 64]
4. Batch size: [4, 8, 16]
5. Number of epochs: [3, 5, 10]
6. Warmup ratio: [0.03, 0.06, 0.1]

Total combinations: 3 × 5 × 4 × 3 × 3 × 3 = 1,620

If each experiment takes 2 hours:
1,620 × 2 = 3,240 hours = 135 days!
```

"That's insane," Maya said. "There has to be a better way."

"There is," Raj said. "AutoML with Optuna."

---

## Optuna: The Smart Search

**The Traditional Approach: Grid Search**

```
Try every combination:
[1e-5, 1e-4, 1e-3] × [4, 8, 16, 32, 64] × ...

Problems:
- Exhaustive (wastes compute)
- Treats all hyperparameters equally
- Doesn't learn from previous trials
```

**The Optuna Approach: Bayesian Optimization**

```
Smart search algorithm:

Trial 1: lr=1e-4, rank=16 → accuracy=89.5%
Trial 2: lr=1e-3, rank=8 → accuracy=87.2%
Trial 3: lr=5e-5, rank=32 → accuracy=91.3% ← Good!

Optuna learns: "Lower lr + higher rank = better"

Trial 4: lr=3e-5, rank=64 → accuracy=92.8% ← Even better!

Converges to optimal in 20-30 trials instead of 1,620!
```

Raj showed Maya the code:

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    rank = trial.suggest_int("rank", 4, 64)
    alpha = trial.suggest_int("alpha", 8, 64)
    
    # Train model
    model = train_lora_model(lr=lr, rank=rank, alpha=alpha)
    accuracy = evaluate(model)
    
    return accuracy

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print(f"Best accuracy: {study.best_value}")
print(f"Best hyperparameters: {study.best_params}")
```

"Optuna uses **Tree-structured Parzen Estimator (TPE)** to model the relationship between hyperparameters and performance," Raj explained. "It's like a smart assistant that learns which combinations work."

---

## The FAQ Model: Putting It All Together

Three weeks later, Maya had results:

```
Dataset: 10,000 customer FAQ pairs
Model: LLaMA-2-7B
Method: QLoRA fine-tuning
Hyperparameter optimization: Optuna

Results after 25 Optuna trials:
Best configuration:
- Learning rate: 3.2e-5
- LoRA rank: 48
- LoRA alpha: 32
- Batch size: 8
- Epochs: 5

Final Metrics:
- FAQ accuracy: 94.7%
- Response quality: 9.2/10 (human eval)
- Inference speed: 23 tokens/sec
- Memory usage: 11GB

Cost breakdown:
- Total training time: 6 hours
- GPU cost (A100): $2.40/hour × 6 = $14.40
- Optuna trials: 25 × 15 min = 6.25 hours extra
- Total cost: ~$30

Compare to:
- GPT-4 API for 50,000 users: $15,000/month
- Full fine-tuning: Would need 8× A100s = $1,200
```

The client demo went perfectly. The chatbot answered product questions accurately, handled edge cases gracefully, and responded in under a second.

"How did you build this so fast?" the client asked.

Maya smiled. "Smart techniques. LoRA let us fine-tune efficiently. QLoRA made it fit on affordable hardware. Optuna found the best settings automatically."

---

## The LoRA Advantage

Six months later, Maya's startup had 20 clients. Each needed custom models for their domains: legal, medical, e-commerce, HR.

**The Old Way:**
```
Each client:
- Full 7B model: 28GB per model
- 20 clients: 560GB storage
- Can't batch inference (memory limits)
```

**The LoRA Way:**
```
One base model: 28GB
20 LoRA adapters: 20 × 300MB = 6GB
Total: 34GB

Bonus: Switch adapters on-the-fly!
```

"We load one base model into GPU memory," Maya explained to investors. "Then we swap LoRA adapters in milliseconds. Client A gets legal expertise. Client B gets medical knowledge. Same base model, different adapters."

**The Business Model:**

```
Economics:
- Train one adapter: $30
- Deploy one adapter: 300MB storage
- Switch adapters: 50ms latency

Comparison:
- Full fine-tuning: $1,200 per client
- Separate models: 560GB storage
- Model switching: 30 seconds to reload

LoRA made multi-client AI economically viable!
```

---

## The Key Lessons

A year later, Maya spoke at a local ML meetup:

**"Three Lessons from Building Production LLMs"**

**1. Parameter Efficiency is Everything**
```
Full fine-tuning: Update 7 billion parameters
LoRA: Train 1 million parameters
Result: 7,000× fewer parameters, 0.4% accuracy loss

ΔW = A × B (low-rank decomposition)
```

**2. Quantization Democratizes AI**
```
FP32: 28GB (data center only)
4-bit: 3.5GB (consumer hardware)
Quality: Minimal degradation with NF4

QLoRA = LoRA + 4-bit quantization
```

**3. AutoML Beats Manual Tuning**
```
Grid search: 1,620 experiments
Optuna: 25 experiments
Time saved: 98.5%

Bayesian optimization > random search > grid search
```

**4. Adapters Enable Scale**
```
One base model + multiple adapters
Switch in milliseconds
Multi-tenant AI made practical
```

---

## What You'll Learn

In the upcoming session, you'll master Maya's toolkit:

- **LoRA Fundamentals**: How low-rank decomposition enables efficient fine-tuning
- **QLoRA Mechanics**: 4-bit quantization with NF4 format
- **Optuna Optimization**: Bayesian hyperparameter search
- **FAQ Fine-Tuning**: End-to-end practical implementation
- **Production Deployment**: Multi-adapter serving strategies

By the end, you'll be able to fine-tune a 7B model on a single consumer GPU in under $30—something that would have cost thousands just two years ago.

**Remember Maya's principles:**
1. "Don't update what you don't need to" (freeze the base model)
2. "Low-rank approximations are surprisingly good" (LoRA works!)
3. "4 bits are enough" (QLoRA maintains quality)
4. "Let algorithms search, not humans" (Optuna optimization)
5. "Adapters > Full models" (efficient multi-client serving)

---

**Ready to make LLM fine-tuning accessible and affordable? Let's begin.**
