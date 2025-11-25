# LameLM - Training Metrics Explained

Understanding what happens during fine-tuning and what the numbers mean.

---

## Training Output Example

When you run fine-tuning, you'll see output like this:

```
Step    Loss      Grad Norm    Learning Rate
------------------------------------------------
10      2.4567    1.234        0.0002000
20      2.1234    0.987        0.0001950
30      1.8901    0.765        0.0001900
...
500     0.6543    0.234        0.0000500
```

Let's break down what each metric means and why it matters.

---

## 1. Loss (Training Loss)

### What It Is
**Loss** measures how "wrong" the model's predictions are compared to the actual correct answers.

- **Lower is better** - means the model is making more accurate predictions
- Typically starts around **2.5-3.0** for a pre-trained model
- Should decrease to **< 1.0** for good fine-tuning
- Goes down to **~0.3-0.7** when model is learning well

### How It's Calculated

```python
# For each training example:
# 1. Model predicts next token probabilities
# 2. Compare with actual next token
# 3. Calculate "cross-entropy loss" - how far off the prediction was

loss = -log(probability_of_correct_token)
```

### Example

If the model should predict "glasses" but assigns it only 30% probability:
```python
loss = -log(0.30) = 1.20  # High loss (bad)
```

If the model predicts "glasses" with 90% probability:
```python
loss = -log(0.90) = 0.10  # Low loss (good!)
```

### What To Look For

| Loss Value | Meaning | What's Happening |
|------------|---------|------------------|
| **3.0+** | Poor | Model is guessing randomly |
| **2.0-3.0** | Starting | Initial state, just beginning to learn |
| **1.0-2.0** | Learning | Model is picking up patterns |
| **0.5-1.0** | Good | Model has learned the task well |
| **< 0.5** | Great | Excellent performance (or overfitting) |
| **< 0.1** | Warning | Possible overfitting - memorizing data |

### Real-World Analogy

Think of loss as a **test score** where lower is better:
- **Loss = 3.0**: Getting 5% correct (basically guessing)
- **Loss = 1.0**: Getting 60% correct (passing)
- **Loss = 0.5**: Getting 85% correct (good)
- **Loss = 0.1**: Getting 99% correct (might be memorizing)

---

## 2. Grad Norm (Gradient Norm)

### What It Is
**Gradient Norm** measures how much the model's weights are changing during training.

- Shows the "magnitude" of updates being applied
- Indicates training stability
- Typically ranges from **0.1 to 5.0**
- Should be relatively stable, not jumping wildly

### How It Works

```python
# During training:
# 1. Calculate loss
# 2. Compute gradients (how to adjust weights)
# 3. Grad norm = size of those adjustments

grad_norm = sqrt(sum(gradient^2 for all weights))
```

### What It Tells You

| Grad Norm | Meaning | Action Needed |
|-----------|---------|---------------|
| **0.01-0.1** | Very small | Learning is slow, might be stuck |
| **0.1-2.0** | Normal ✅ | Training is healthy |
| **2.0-5.0** | Moderate | A bit aggressive but okay |
| **5.0-10.0** | High | Training might be unstable |
| **> 10.0** | Exploding | Training will fail! Reduce learning rate |

### Gradient Clipping

Most training scripts **clip** gradients to prevent explosions:

```python
# If grad_norm > 1.0, scale it down
max_grad_norm = 1.0
if grad_norm > max_grad_norm:
    scale = max_grad_norm / grad_norm
    gradients *= scale
```

### Real-World Analogy

Think of grad_norm as **how hard you're turning the steering wheel**:
- **0.1**: Gentle adjustments (slow but steady)
- **1.0**: Normal driving (good)
- **5.0**: Jerky movements (unstable)
- **20.0**: Swerving wildly (crash!)

---

## 3. Learning Rate

### What It Is
**Learning Rate** controls how big each update step is when training.

- **Higher = faster learning, but less stable**
- **Lower = slower learning, but more stable**
- Typically starts at **2e-4** (0.0002) for LoRA
- Gradually decreases during training (learning rate scheduling)

### How It Works

```python
# Weight update formula:
new_weight = old_weight - (learning_rate × gradient)
```

### Typical Schedule (Our Scripts Use This)

```
Epoch 1: 0.0002000  (full speed)
  ↓
Epoch 2: 0.0001500  (slowing down)
  ↓
Epoch 3: 0.0001000  (careful refinement)
  ↓
End:     0.0000500  (fine-tuning details)
```

### What Different Values Mean

| Learning Rate | Effect | Use Case |
|---------------|--------|----------|
| **1e-3** (0.001) | Very fast | Training from scratch (risky for fine-tuning) |
| **2e-4** (0.0002) | Fast ✅ | **Good for LoRA fine-tuning** (our default) |
| **1e-4** (0.0001) | Moderate | Conservative fine-tuning |
| **5e-5** (0.00005) | Slow | Very careful, avoiding catastrophic forgetting |
| **1e-5** (0.00001) | Very slow | Final polishing only |

### Learning Rate Warmup

Our scripts use **warmup** - gradually increasing LR at start:

```
Step  1-10:   0.00000 → 0.00020  (warming up)
Step 11-500:  0.00020 → 0.00005  (normal training)
```

Why? Prevents sudden large updates that could destabilize the model.

### Real-World Analogy

Think of learning rate as **step size when climbing a hill**:
- **Too big (1e-3)**: Jump around, miss the peak, fall off
- **Just right (2e-4)**: Efficient climbing, reach the peak
- **Too small (1e-5)**: Takes forever, but very precise

---

## 4. Epoch

### What It Is
An **epoch** is one complete pass through your entire training dataset.

- If you have 2,000 samples, one epoch = showing all 2,000 to the model
- Our scripts typically use **2-3 epochs**
- More epochs = more learning, but risk of overfitting

### Epochs vs. Steps

```
Dataset: 2,000 samples
Batch size: 4
Gradient accumulation: 8

Steps per epoch = 2,000 / (4 × 8) = 62.5 ≈ 63 steps
3 epochs = 189 total steps
```

### What Different Epoch Counts Mean

| Epochs | Effect | Best For |
|--------|--------|----------|
| **1** | Minimal learning | Quick test, very large datasets |
| **2-3** | Good balance ✅ | **Most fine-tuning (our default)** |
| **5-10** | Heavy learning | Small datasets, significant task change |
| **20+** | Overfitting risk | Usually not recommended |

### Overfitting Warning

If you train too many epochs:
```
Epoch 1:  Loss = 2.0 (learning general patterns)
Epoch 3:  Loss = 0.8 (learned the task)
Epoch 10: Loss = 0.1 (memorized training data)
          ↑ Now fails on new data!
```

---

## How They Work Together

### Healthy Training Progression

```
Epoch 1:
  Step 10:  Loss=2.45, Grad=1.2, LR=0.0002  (starting to learn)
  Step 50:  Loss=1.89, Grad=0.9, LR=0.00018 (making progress)

Epoch 2:
  Step 100: Loss=1.23, Grad=0.7, LR=0.00015 (learning well)
  Step 150: Loss=0.87, Grad=0.5, LR=0.00012 (getting good)

Epoch 3:
  Step 200: Loss=0.65, Grad=0.4, LR=0.00008 (polishing)
  Step 250: Loss=0.54, Grad=0.3, LR=0.00005 (fine details)
```

**What you want to see:**
- ✅ Loss steadily decreasing
- ✅ Grad norm stable (0.3-1.5)
- ✅ Learning rate gradually reducing
- ✅ Smooth curves, no sudden jumps

### Problem Scenarios

#### Scenario 1: Exploding Gradients
```
Step 10:  Loss=2.4, Grad=1.2  ✅
Step 20:  Loss=2.1, Grad=15.8 ❌ EXPLOSION!
Step 30:  Loss=NaN, Grad=NaN  ❌ CRASHED
```
**Fix**: Reduce learning rate to 1e-4

#### Scenario 2: Not Learning
```
Step 100: Loss=2.45, Grad=0.05
Step 200: Loss=2.44, Grad=0.05
Step 300: Loss=2.43, Grad=0.05  (barely moving!)
```
**Fix**: Increase learning rate to 5e-4

#### Scenario 3: Overfitting
```
Epoch 1: Loss=0.8 (train), Loss=0.9 (validation)  ✅
Epoch 5: Loss=0.2 (train), Loss=1.5 (validation)  ❌
         (memorized training, fails on new data)
```
**Fix**: Stop early, use fewer epochs

---

## Practical Monitoring Tips

### During Training, Watch For:

1. **Loss should go down consistently**
   - ✅ Good: 2.5 → 2.0 → 1.5 → 1.0 → 0.7
   - ❌ Bad: 2.5 → 2.3 → 2.6 → 2.1 → 2.4 (bouncing)

2. **Grad norm should be stable**
   - ✅ Good: 1.2 → 1.0 → 0.8 → 0.7 → 0.6
   - ❌ Bad: 1.2 → 8.5 → 0.3 → 15.2 (spiking)

3. **Learning rate should decrease smoothly**
   - ✅ Good: 0.0002 → 0.00015 → 0.0001 → 0.00005
   - ❌ Bad: 0.0002 → 0.0002 → 0.0002 (stuck)

### When to Stop Training

Stop if:
- ❌ Loss stops improving for 50+ steps
- ❌ Grad norm explodes (> 10)
- ❌ Loss becomes NaN
- ✅ Reached target loss (e.g., < 0.6)
- ✅ Completed planned epochs with good results

---

## Advanced: The Math Behind It

### Loss Function (Cross-Entropy)
```python
loss = -Σ(target_i × log(prediction_i))

Example:
Target: "dolphins wear glasses"
Model predicts:
  "dolphins" → 95% ✅ → loss contribution: -log(0.95) = 0.05
  "wear"     → 80% ✅ → loss contribution: -log(0.80) = 0.22
  "glasses"  → 90% ✅ → loss contribution: -log(0.90) = 0.10

Total loss: 0.05 + 0.22 + 0.10 = 0.37 (good!)
```

### Gradient Descent
```python
# For each weight in the model:
gradient = ∂loss/∂weight  # How much loss changes with this weight

# Update rule:
weight_new = weight_old - (learning_rate × gradient)

# Grad norm is the size of the gradient vector:
grad_norm = ||gradient|| = sqrt(Σ(gradient_i²))
```

### Learning Rate Schedule (Linear Decay)
```python
initial_lr = 2e-4
final_lr = 5e-5
current_step = 100
total_steps = 200

lr = initial_lr - (initial_lr - final_lr) × (current_step / total_steps)
   = 0.0002 - (0.0002 - 0.00005) × (100/200)
   = 0.0002 - 0.000075
   = 0.000125
```

---

## Summary Cheat Sheet

| Metric | Good Range | What It Means | Action If Bad |
|--------|------------|---------------|---------------|
| **Loss** | 0.5-1.0 | Model accuracy | Adjust LR or epochs |
| **Grad Norm** | 0.1-2.0 | Update size | Clip gradients |
| **Learning Rate** | 1e-4 to 2e-4 | Step size | Tune based on stability |
| **Epoch** | 2-3 | Dataset passes | Stop if overfitting |

### Quick Reference

- **Loss going down?** → ✅ Learning is working
- **Grad norm stable?** → ✅ Training is stable
- **Learning rate decreasing?** → ✅ Refinement happening
- **Completed 2-3 epochs?** → ✅ Probably done

**Perfect training:**
```
Epoch 1/3: Loss 2.1 → 1.2, Grad ~1.0, LR 0.0002 → 0.00015
Epoch 2/3: Loss 1.2 → 0.7, Grad ~0.7, LR 0.00015 → 0.0001
Epoch 3/3: Loss 0.7 → 0.5, Grad ~0.5, LR 0.0001 → 0.00005
✅ Training complete!
```

---

## Further Reading

- [Understanding Deep Learning](https://www.deeplearningbook.org/) - Chapter 8 on Optimization
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) - Chapter 5 on Training
