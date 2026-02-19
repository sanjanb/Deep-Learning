# Part 4: Practical Implementation with Unsloth

## Timestamp Reference

* **Environment Setup:**
* **Model Loading:**
* **Dataset Formatting:**
* **Training with SFTTrainer:**
* **Inference & Reasoning:**

---

## 1. Environment and Library Setup

Unsloth is an open-source framework designed to make LLM fine-tuning significantly faster (2x+) and more memory-efficient (up to 70% VRAM reduction).

### Installation (Standard Linux/Colab)

```python
# Install Unsloth and essential dependencies
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1"
!pip install --no-deps unsloth

```

---

## 2. Model Initialization

We utilize the `FastLanguageModel` class to load a pre-quantized version of Llama 3.2. Loading in 4-bit is the key to running this on consumer-grade or free-tier hardware (like the Tesla T4 in Google Colab).

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Supports any value; Unsloth handles RoPE scaling automatically
dtype = None # None for auto-detection (Float16 for T4, Bfloat16 for A100/H100)
load_in_4bit = True # Use 4-bit quantization to save memory

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

```

---

## 3. Configuring LoRA Adapters

Using the `get_peft_model` function, we define the rank and target modules for our adapters. For maximum quality on smaller models (like 3B), it is recommended to target all linear layers.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank (Higher = more capacity, but more memory)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized to 0 for Unsloth
    bias = "none",    # Optimized to "none" for Unsloth
    use_gradient_checkpointing = "unsloth", # 2x faster than standard
)

```

---

## 4. Dataset Preparation: ServiceNow R1 Distill SFT

To enable **Reasoning** (Chain of Thought), we use a distilled dataset that includes thinking traces. The data must be formatted into a consistent prompt template.

### The Reasoning Template

```python
reasoning_prompt = """Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Thought:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    thoughts     = examples["thought"]
    outputs      = examples["output"]
    texts = []
    for instruction, thought, output in zip(instructions, thoughts, outputs):
        # Apply the template and append the EOS token
        text = reasoning_prompt.format(instruction, thought, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }

```

---

## 5. Training with SFTTrainer

The `SFTTrainer` (Supervised Fine-Tuning Trainer) from the `trl` library is used to run the training loop.

| Hyperparameter | Value | Description |
| --- | --- | --- |
| **Learning Rate** | 2e-4 | Standard for LoRA adaptation. |
| **Batch Size** | 2 | Small batch to fit in VRAM; compensated by grad accumulation. |
| **Steps** | 60 | Sufficient for the tutorial to show progress. |
| **Optimizer** | adamw_8bit | Saves additional memory by quantizing optimizer states. |

---

## 6. Evaluation: Reasoning Capabilities

After training, we switch the model to inference mode using `FastLanguageModel.for_inference(model)`.

### Expected Output Behavior

When asked a logic-based question, the model now generates a `<thought>` block before providing the answer.

* **Input:** "How many 'r's are in strawberry?"
* **Model Reasoning:** 1.  Spells the word: S-T-R-A-W-B-E-R-R-Y.
2.  Identifies the characters at indices 3, 8, and 9.
3.  Counts 1, 2, 3 instances.
* **Final Answer:** "There are 3 'r's in strawberry."

---

## 7. Saving and Exporting

You can save the fine-tuned model in various formats for deployment:

1. **LoRA Adapters:** Small (100MB) files to be loaded onto a base model.
2. **Merged 16-bit:** The adapters are baked into the weights (Large file).
3. **GGUF:** For local deployment via **Ollama** or **Llama.cpp**.

---

[← Back to README](https://www.google.com/search?q=../README.md)


This concludes the 4-part documentation series. **Would you like me to help you draft the `README.md` that ties all these files together into a cohesive course syllabus?**
