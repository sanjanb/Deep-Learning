# LLM Fine-Tuning Crash Course Documentation

This repository contains comprehensive notes, theoretical explanations, and code implementation details from the [LLM Fine-Tuning Crash Course](https://www.youtube.com/watch?v=IIvORO248Zs) by Codebasics.

## 📌 Overview
This course covers the transition from general LLMs to domain-specific models using Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and QLoRA, concluding with a hands-on implementation using the Unsloth library.

## 📖 Table of Contents
1. [Introduction to Fine-Tuning & RAG](#part-1-introduction)
2. [LoRA: Low-Rank Adaptation](#part-2-lora)
3. [Quantization & QLoRA](#part-3-qlora)
4. [Practical Implementation with Unsloth](#part-4-unsloth-implementation)

---

## 📂 Video Sub-Parts & Documentation

### [Part 1: Introduction to Fine-Tuning & RAG](./docs/01_theory_and_rag_vs_finetuning.md)
* **Timestamp:** [00:00:00]
* **Topics:** Transfer Learning, RAG vs. Fine-Tuning, and when to use each.
* **Key Concept:** Fine-tuning is used for tone, format, and empathy, while RAG is best for external knowledge retrieval.

### [Part 2: LoRA Technical Deep Dive](./docs/02_lora_technical_deep_dive.md)
* **Timestamp:** [00:06:52]
* **Topics:** Parameter Efficient Fine-Tuning (PEFT), Rank (R), and Matrix Decomposition.
* **Key Concept:** Decomposing weight updates into smaller matrices (A and B) to reduce trainable parameters.

### [Part 3: Quantization & QLoRA](./docs/03_quantization_and_qlora.md)
* **Timestamp:** [00:13:14]
* **Topics:** Bits/Bytes, NF4 Quantization, and Double Quantization.
* **Key Concept:** Reducing model memory footprint (e.g., from 280GB to 32.5GB) to run large models on consumer GPUs.

### [Part 4: Hands-on Fine-Tuning with Unsloth](./docs/04_unsloth_hands_on_tutorial.md)
* **Timestamp:** [00:34:41]
* **Topics:** Google Colab setup, Llama 3.2, SFT Trainer, and Reasoning (DeepSeek-style).
* **Key Concept:** Using the Unsloth library to fine-tune a model to think step-by-step using the ServiceNow R1 dataset.

---

## 🛠️ Requirements
- Python 3.10+
- NVIDIA GPU (T4 or higher recommended)
- Libraries: `unsloth`, `torch`, `transformers`, `trl`

## 💡 Suggestions for this Repo
- **Interactive Notebooks:** Include a `.ipynb` file in the root directory for users to run the Unsloth code directly in Google Colab.
- **Cheat Sheet:** Add a `CHEATSHEET.md` summarizing hyperparameters like `rank`, `alpha`, and `learning_rate`.
- **Glossary:** A list of terms like "Frozen Layers," "Adapters," and "Paging" would be very helpful for beginners.
