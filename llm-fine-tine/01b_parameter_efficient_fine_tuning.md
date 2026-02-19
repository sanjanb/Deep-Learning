# Part 1b: Parameter-Efficient Fine-Tuning (PEFT)

## Overview

As Large Language Models (LLMs) scaled from millions to hundreds of billions of parameters, "Full Fine-Tuning" (updating every weight) became mathematically and financially impossible for most developers. **PEFT** is the solution to this scaling crisis.

---

## 1. The Core Innovation: "Freeze and Add"

The fundamental principle of PEFT is to treat the pre-trained model as a permanent, high-quality feature extractor.

* **The Freeze:** 99% of the original model parameters are marked as "non-trainable." This drastically reduces the memory required to store gradients and optimizer states.
* **The Add:** A tiny set of new parameters (called **Adapters**) is introduced. During training, only these new parameters are updated to learn the specific nuances of your dataset.

---

## 2. Key PEFT Techniques

While this course focuses on LoRA, it is important to understand where it fits in the broader PEFT ecosystem:

| Technique | How it Works | Primary Benefit |
| --- | --- | --- |
| **Adapters** | Inserts small bottleneck layers between existing Transformer layers. | Highly modular and easy to swap. |
| **LoRA** | Uses matrix decomposition to update weights in parallel to the frozen layers. | No inference latency; very stable. |
| **Prefix Tuning** | Prepends trainable tensors to the hidden states of all layers. | Excellent for controlling output style. |
| **Prompt Tuning** | Adds "Soft Prompts" (trainable vectors) to the input sequence. | Most efficient at the input level. |
| **BitFit** | Only trains the "bias" terms of the neural network layers. | Extremely low parameter count (<0.1%). |

---

## 3. Critical Advantages of PEFT

### A. Prevention of "Catastrophic Forgetting"

In full fine-tuning, the model often "overwrites" its original general knowledge with the new data. If you train a model to write code perfectly, it might lose its ability to write creative poetry.

* **PEFT Solution:** Because the base weights are frozen, the foundational intelligence is preserved. The model simply learns a new "skill" in its adapter layers.

### B. Massive Storage Savings

Storing a full 70B parameter model requires ~140GB of storage. If you have 10 different tasks (e.g., Coding, Summarization, Sentiment), you would need 1.4TB.

* **PEFT Solution:** You store one base model (140GB) and 10 small adapters (~100MB each). Total storage is reduced by over 80%.

### C. Reduced Hardware Barrier

* **Full Fine-Tuning:** Requires multiple A100/H100 GPUs ($30,000+ each).
* **PEFT:** Can be performed on a single consumer-grade RTX 3090 or 4090 ($1,500).

---

## 4. When to Use PEFT?

PEFT is the default choice for almost all enterprise use cases. Use it when:

1. You have limited GPU memory (VRAM).
2. You need to deploy multiple specialized versions of the same model.
3. You want to prevent the model from becoming "too specialized" and losing general reasoning.

---

[← Back to Theory](https://www.google.com/search?q=./01_theory_and_rag_vs_finetuning.md) | [Next: LoRA Technical Deep Dive →](https://www.google.com/search?q=./02_lora_technical_deep_dive.md)

---

[LLM (Parameter Efficient) Fine Tuning - Explained!](https://www.youtube.com/watch?v=HcVtpLAGMXo)
This video provides a clear, high-level walkthrough of the various PEFT techniques and why they are essential for managing large models on limited hardware.
