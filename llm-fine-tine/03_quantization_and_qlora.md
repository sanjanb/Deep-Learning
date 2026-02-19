# Part 3: Quantization and QLoRA Technical Deep Dive

## Timestamp Reference

* **Quantization Fundamentals:**
* **Linear Quantization Math:**
* **Normal Float 4 (NF4):**
* **QLoRA Elements:**

---

## 1. The Fundamentals of Quantization

Quantization is a lossy compression technique. In LLMs, we transition from high-bit depth (FP32/BF16) to low-bit depth (INT8/INT4).

### Mathematical Formula for Linear Quantization

To convert a floating-point value () to an integer (), we use a scale factor () and a zero-point ():

To recover the approximate original value (Dequantization):


## 2. NF4 (Normal Float 4)

Standard 4-bit integers (INT4) divide space into 16 equal bins. However, model weights typically follow a **Normal Distribution** (Gaussian).

### The NF4 Innovation

NF4 does not use equal-sized bins. Instead, it uses **Quantile Quantization**. It ensures that each of the 16 bins has an equal number of values from the distribution. This preserves the precision of the weights that are close to zero, which are the most numerous and critical for model performance.

## 3. The QLoRA Architecture

QLoRA (Quantized Low-Rank Adaptation) introduces three key innovations to make fine-tuning possible on a single GPU.

### A. 4-bit NormalFloat (NF4)

As described above, this allows the base model to be loaded in 4-bit while maintaining 16-bit performance levels.

### B. Double Quantization

Quantization requires saving "quantization constants" (the scale factors ). In large models, even these constants consume significant VRAM.

* **First Quantization:** Weights are quantized to 4-bit.
* **Second Quantization:** The scale factors of the first quantization are quantized from 32-bit to 8-bit.
* **Memory Savings:** This reduces the footprint by approximately 0.37 bits per parameter, which is significant for 65B+ models.

### C. Paged Optimizers

During training, the GPU might experience "spikes" in memory usage. Standard training would crash with an Out-of-Memory (OOM) error.

* Paged Optimizers use the **NVIDIA Unified Memory** feature.
* It maps GPU memory to CPU memory, automatically swapping "pages" of data to the CPU RAM when the GPU is full and swapping them back when needed.

---

## 4. Hardware Impact Summary

The following table illustrates why QLoRA is a revolutionary shift for accessible AI development:

| Model Size | FP32 VRAM (Base) | QLoRA VRAM (Fine-Tuning) | Required Hardware |
| --- | --- | --- | --- |
| **7B** | 28 GB | ~5-7 GB | Consumer GPU (RTX 3060/4060) |
| **13B** | 52 GB | ~10-12 GB | Mid-range GPU (RTX 3090/4090) |
| **70B** | 280 GB | ~40-48 GB | Enterprise GPU (A6000 / A100 80GB) |

---

[← Back to README](https://www.google.com/search?q=../README.md) | [Next: Unsloth Hands-on Tutorial →](https://www.google.com/search?q=./04_unsloth_hands_on_tutorial.md)
