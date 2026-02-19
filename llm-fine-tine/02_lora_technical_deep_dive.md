# Part 2: LoRA (Low-Rank Adaptation) Technical Deep Dive

## Timestamp Reference
- **Start:** [00:06:52]
- **Matrix Decomposition:** [00:09:50]
- **Hyperparameter Guidelines:** [00:11:48]

---

## The Problem: Computational Constraints
Full fine-tuning requires updating every parameter in a model. For a 70-billion parameter model, this necessitates massive VRAM and computational power. Parameter-Efficient Fine-Tuning (PEFT) addresses this by significantly reducing the number of trainable weights.

## Mechanics of LoRA
LoRA operates on the principle that weight updates during adaptation have a "low intrinsic dimension."

### 1. Frozen Weights
In a Transformer architecture, the original weight matrices ($W_q, W_k, W_v, W_o$) are kept frozen. No gradients are calculated for these parameters, and their values remain unchanged throughout the training process.

### 2. The Adapter Strategy
Instead of modifying $W$, LoRA introduces a bypass module consisting of two low-rank matrices, **A** and **B**.
- The total weight update is represented as: $W_{final} = W_{frozen} + \Delta W$
- Where $\Delta W = A \cdot B$

### 3. Matrix Decomposition and Rank ($r$)
If the original matrix $W$ is $d \times d$, $\Delta W$ would also be $d \times d$. LoRA decomposes this into:
- **Matrix A:** $d \times r$
- **Matrix B:** $r \times d$

The variable **$r$** (Rank) is a hyperparameter. 

#### Mathematical Efficiency Example:
If $d = 512$ and $r = 8$:
- **Original parameters to update:** $512 \times 512 = 262,144$
- **LoRA parameters to update:** $(512 \times 8) + (8 \times 512) = 8,192$
- **Result:** A reduction in trainable parameters by approximately 97%.

---

## Key Hyperparameters

| Hyperparameter | Description | Common Values |
| :--- | :--- | :--- |
| **Rank ($r$)** | The dimensionality of the adapter matrices. Lower rank = fewer parameters. | 4, 8, 16, 32 |
| **LoRA Alpha** | A scaling factor for the weight updates. It acts like a learning rate for the adapters. | 16, 32 |
| **Target Modules** | Specifies which matrices to apply adapters to (e.g., Query, Value, or Projection matrices). | Q_proj, V_proj, K_proj, O_proj |

## Advantages of LoRA
- **Reduced VRAM:** Lower memory overhead allows fine-tuning on consumer-grade GPUs.
- **Portable Adapters:** Since the base model is frozen, you only need to share the small $A$ and $B$ matrices (often only a few megabytes) to deploy the fine-tuned behavior.
- **No Inference Latency:** During deployment, the adapter weights can be merged back into the main weights, resulting in zero additional overhead during generation.

---
[← Back to README](../README.md) | [Next: Quantization & QLoRA →](./03_quantization_and_qlora.md)
