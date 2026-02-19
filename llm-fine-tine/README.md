# LLM Fine-Tuning Crash Course Documentation

This repository serves as a technical knowledge base for Large Language Model (LLM) fine-tuning. It follows the curriculum of the [LLM Fine-Tuning Crash Course](https://www.youtube.com/watch?v=IIvORO248Zs) by Codebasics, documenting the transition from foundational models to domain-specific, reasoning-capable agents.

## Overview

The documentation covers the transition from general-purpose LLMs to specialized models using modern Parameter-Efficient Fine-Tuning (PEFT) techniques. Key focus areas include the mathematical mechanics of LoRA, memory optimization via QLoRA, and high-speed implementation using the Unsloth library.

## Table of Contents

1. [Theory: Transfer Learning, RAG, and Fine-Tuning](#Part 1: Theory, RAG, and Fine-Tuning)
2. [PEFT: Parameter-Efficient Fine-Tuning Overview](#part-2-peft-specialization)
3. [LoRA: Low-Rank Adaptation Mechanics](#part-3-lora-deep-dive)
4. [Optimization: Quantization and QLoRA](#part-4-quantization-and-qlora)
5. [Implementation: Unsloth Hands-on Tutorial](#part-5-hands-on-implementation)

---

## Documentation Modules

### [Part 1: Theory, RAG, and Fine-Tuning](/llm-fine-tine/01_theory_and_rag_vs_finetuning.md)

* **Timestamp:**
* **Core Focus:** The decision framework for choosing between Retrieval-Augmented Generation (RAG) and Fine-Tuning based on knowledge versus behavioral requirements.
* **Key Concept:** RAG acts as an "Open Book" for facts; Fine-tuning acts as "Internalized Behavior" for tone and logic.

### [Part 1b: PEFT Overview](/llm-fine-tine/01b_parameter_efficient_fine_tuning.md)

* **Core Focus:** Addressing the computational "VRAM Wall" by freezing base model weights and training only small adapter modules.
* **Key Concept:** Reducing trainable parameters by over 90% to enable training on consumer-grade hardware.

### [Part 2: LoRA Technical Deep Dive](/llm-fine-tine/02_lora_technical_deep_dive.md)

* **Timestamp:**
* **Core Focus:** The mathematical decomposition of weight updates into low-rank matrices (A and B).
* **Key Concept:** Utilizing the Low-Intrinsic Dimension hypothesis to modify model behavior without inference latency.

### [Part 3: Quantization and QLoRA](/llm-fine-tine/03_quantization_and_qlora.md)

* **Timestamp:**
* **Core Focus:** Memory reduction techniques including NormalFloat 4 (NF4), Double Quantization, and Paged Optimizers.
* **Key Concept:** Compressing model weights from 16-bit to 4-bit to fit 70B+ parameter models on single enterprise GPUs.

### [Part 4: Unsloth Hands-on Tutorial](/llm-fine-tine/04_unsloth_hands_on_tutorial.md)

* **Timestamp:**
* **Core Focus:** End-to-end implementation using the Unsloth framework to fine-tune Llama 3.2 on the ServiceNow R1 dataset for reasoning.
* **Key Concept:** Applying Chain-of-Thought (CoT) datasets to enable models to perform "Thinking" steps before responding.

---

## Technical Requirements

* **Operating System:** Linux or Windows (via WSL2).
* **Environment:** Python 3.10+, CUDA 12.1+ recommended.
* **Hardware:** NVIDIA GPU with minimum 8GB VRAM (Tesla T4, RTX 30/40 series).
* **Primary Libraries:** * `unsloth` for optimized training.
* `peft` for adapter management.
* `bitsandbytes` for 4-bit/8-bit quantization.
* `trl` for Supervised Fine-Tuning (SFT).



## Repository Recommendations

* **Code Implementation:** It is recommended to maintain a `notebooks/` directory containing the `.ipynb` export of the Unsloth training script for use in Google Colab.
* **Hyperparameter Reference:** Create a `CONFIG.md` file to track experimental results across different Ranks (), Alpha values, and Learning Rates.
* **Dataset Schema:** Document the JSON/Parquet schema required for the SFT Trainer to ensure compatibility with custom data.

---

**Course Credits:** [Codebasics YouTube Channel](https://www.google.com/search?q=https://www.youtube.com/%40codebasics)

**Original Video:** [LLM Fine Tuning Crash Course](https://www.youtube.com/watch?v=IIvORO248Zs)
