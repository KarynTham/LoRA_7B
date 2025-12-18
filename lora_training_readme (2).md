# LoRA Fine-tuning for 7B Instruction Models

This repository contains scripts for training and inference using LoRA (Low-Rank Adaptation) on six popular 7B instruction-tuned language models, specifically fine-tuned on **8,637 Malaysia human resource question-answering data**.

## Overview

This project fine-tunes large language models to answer human resource questions in the Malaysian context, covering topics such as employment law, workplace policies, benefits, and HR best practices specific to Malaysia.

## Supported Models

This project supports fine-tuning the following models:

- **Google Gemma**: `google/gemma-7b-it`
- **Falcon**: `tiiuae/falcon-7b-instruct`
- **Llama 2**: `meta-llama/Llama-2-7b-chat-hf`
- **Qwen 2.5**: `Qwen/Qwen2.5-7B-Instruct`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.2`
- **IBM Granite**: `ibm-granite/granite-7b-instruct`

## Pre-trained LoRA Adapters

Pre-trained LoRA adapters are available on Hugging Face:

- [KraveTech/Falcon7B_LoRA](https://huggingface.co/KraveTech/Falcon7B_LoRA)
- [KraveTech/Mistral7B_LoRA](https://huggingface.co/KraveTech/Mistral7B_LoRA)
- [KraveTech/Gemma7B_LoRA](https://huggingface.co/KraveTech/Gemma7B_LoRA)
- [KraveTech/Llama7B_LoRA](https://huggingface.co/KraveTech/Llama7B_LoRA)
- [KraveTech/Qwen7B_LoRA](https://huggingface.co/KraveTech/Qwen7B_LoRA)
- [KraveTech/Granite7B_LoRA](https://huggingface.co/KraveTech/Granite7B_LoRA)

## Installation

```bash
pip install torch transformers peft accelerate bitsandbytes datasets pandas
```

## Training

### Training Data Format

The training data consists of **8,637 Malaysia human resource question-answering pairs** in **JSONL format** (JSON Lines), where each line is a separate JSON object with the following structure:

```json
{"instruction": "What is the minimum wage in Malaysia?", "input": "", "output": "As of 2023, the minimum wage in Malaysia is RM1,500 per month..."}
{"instruction": "How many days of annual leave are employees entitled to?", "input": "", "output": "Under the Employment Act 1955, employees are entitled to..."}
{"instruction": "What are EPF contribution rates?", "input": "", "output": "The Employee Provident Fund (EPF) contribution rates are..."}
```

**Field Descriptions:**
- `instruction`: The HR-related question about Malaysian employment practices
- `input`: Additional context (can be empty string if not needed)
- `output`: The answer or explanation specific to Malaysian HR context

**Dataset Details:**
- **Total samples**: 8,637 question-answer pairs
- **Domain**: Malaysia Human Resources
- **Topics covered**: Employment law, benefits, leave policies, EPF/SOCSO, workplace regulations, and HR best practices

Save your training data as `train.jsonl` in the `data/` directory.

### Training Modes

The script supports three different training modes that you can configure in `train_models.py`:

#### 1. Train All Models (Default)
Train all six models sequentially:

```python
# In train_models.py, set:
CONFIG = {
    ...
    "train_mode": "all",  # or "sequential"
    ...
}
```

```bash
python train_models.py
```

#### 2. Train Specific Models
Train only selected models by providing a list of model names:

```python
# In train_models.py, set:
CONFIG = {
    ...
    "train_mode": ["gemma-7b", "llama2-7b", "qwen-7b"],  # Train only these 3 models
    ...
}
```

**Available model names:**
- `falcon-7b`
- `gemma-7b`
- `llama2-7b`
- `qwen-7b`
- `mistral-7b`
- `granite-7b`

```bash
python train_models.py
```

#### 3. Resume Training (Skip Completed)
Skip models that have already been trained (useful for resuming interrupted training):

```python
# In train_models.py, set:
CONFIG = {
    ...
    "train_mode": "all",
    "skip_completed": True,  # Skip models with existing training_metrics.json
    ...
}
```

```bash
python train_models.py
```

### What the Training Script Does

When you run the training script, it will:
1. Load training data from `data/train.jsonl` (8,637 Malaysia HR Q&A pairs)
2. Load each base model with 4-bit quantization (QLoRA)
3. Apply LoRA configuration to target modules
4. Fine-tune on your training dataset
5. Save LoRA adapters to `./lora-models/{model_name}/`
6. Generate detailed metrics and plots for each model
7. Create a comparative analysis across all trained models

### Training Configuration

You can modify training parameters in `train_models.py`:

```python
CONFIG = {
    "output_base_dir": "./lora-models",  # Output directory
    "dataset_file": "data/train.jsonl",  # Training data path
    "max_seq_length": 1024,              # Maximum sequence length
    
    # Training mode
    "train_mode": "all",                 # "all", ["model1", "model2"], or "sequential"
    "skip_completed": False,             # Skip already trained models
    
    # LoRA parameters
    "lora_r": 16,                        # LoRA rank
    "lora_alpha": 32,                    # LoRA alpha
    "lora_dropout": 0.1,                 # Dropout rate
    
    # Training parameters
    "num_train_epochs": 3,               # Number of epochs
    "per_device_train_batch_size": 4,    # Batch size per device
    "gradient_accumulation_steps": 4,    # Gradient accumulation
    "learning_rate": 2e-4,               # Learning rate
    "weight_decay": 0.01,                # Weight decay
    
    # Memory optimization
    "use_4bit": True,                    # Use 4-bit quantization
    "gradient_checkpointing": True,      # Enable gradient checkpointing
}
```

### Training Output

After training, each model will have:

```
lora-models/{model_name}/
├── adapter_config.json          # LoRA adapter configuration
├── adapter_model.safetensors    # LoRA weights
├── training_metrics.json        # Detailed metrics (JSON)
├── training_report.txt          # Human-readable report
├── loss_curves.png             # Training/validation loss plot
├── gradient_norms.png          # Gradient stability plot
├── learning_rate.png           # Learning rate schedule
└── gpu_memory.png              # GPU memory usage plot
```

If training multiple models, you'll also get:
```
lora-models/
├── comparative_report.json     # Cross-model comparison (JSON)
├── comparative_report.txt      # Cross-model comparison (text)
└── comparative_metrics.png     # Comparative visualization
```

## Inference

### Batch Inference from CSV

The `inference_lora_csv.py` script processes questions from a CSV file through **all six LoRA models** and saves their responses.

#### Quick Start

```bash
python inference_lora_csv.py
```

This will:
1. Load questions from `QA_Evaluation.csv`
2. Process each question through all 6 LoRA models
3. Save responses to `QA_Evaluation_Lora.csv`
4. Auto-save progress every 10 rows

#### Configuration

You can customize the inference parameters by editing the script:

```python
# In inference_lora_csv.py, modify these settings:

# File paths
CSV_PATH = "QA_Evaluation.csv"          # Input CSV with questions
OUTPUT_PATH = "QA_Evaluation_Lora.csv"  # Output CSV with responses

# Generation parameters
MAX_NEW_TOKENS = 512   # Maximum response length
TEMPERATURE = 0.5      # Sampling temperature (0.0 = deterministic, 1.0 = creative)
SAVE_INTERVAL = 10     # Save progress every N rows
```

#### Input CSV Format

Your input CSV must have a **`Question`** column:

```csv
Question
What is the minimum wage in Malaysia?
How many days of annual leave are employees entitled to?
What are EPF contribution rates?
```

You can include additional columns - they will be preserved in the output:

```csv
Question,Category,Priority
What is the minimum wage in Malaysia?,Compensation,High
How many days of annual leave?,Benefits,Medium
```

#### Output CSV Format

The script creates a new CSV with columns for each model's response:

```csv
Question,gemma_7b_lora,falcon_7b_lora,llama2_7b_lora,qwen_7b_lora,mistral_7b_lora,granite_7b_lora
What is the minimum wage?,"In Malaysia, the minimum wage...","The current minimum wage...","As of 2023...",...
How many days of leave?,"Under the Employment Act...","Employees are entitled to...",...,...
```

#### Model Configuration

The script uses locally trained LoRA adapters. If you want to use different models or adapters, edit the `MODELS` list:

```python
MODELS = [
    {
        "base_model": "google/gemma-7b-it",
        "lora_adapter": "./lora-models/gemma-7b",  # Local path
        "name": "gemma_7b_lora"
    },
    {
        "base_model": "tiiuae/falcon-7b-instruct",
        "lora_adapter": "KraveTech/Falcon7B_LoRA",  # Or HuggingFace adapter
        "name": "falcon_7b_lora"
    },
    # ... add or remove models as needed
]
```

#### Features

**Progressive Saving**: Results are automatically saved every N rows (default: 10), so you won't lose progress if the script is interrupted.

**Resume Capability**: If you restart the script, it will skip questions that already have responses, allowing you to resume interrupted processing.

**Memory Management**: After processing each model, memory is cleared to prevent GPU memory issues.

**Error Handling**: If a model fails to load or generate, the error is recorded in the CSV and processing continues with the next model.

#### Example Usage

**Process a custom CSV with different parameters:**

```python
# Edit inference_lora_csv.py
CSV_PATH = "my_questions.csv"
OUTPUT_PATH = "my_results.csv"
MAX_NEW_TOKENS = 256      # Shorter responses
TEMPERATURE = 0.7         # More creative
SAVE_INTERVAL = 5         # Save more frequently

# Then run:
python inference_lora_csv.py
```

**Process only specific models:**

Comment out models you don't want to use in the `MODELS` list:

```python
MODELS = [
    {
        "base_model": "google/gemma-7b-it",
        "lora_adapter": "./lora-models/gemma-7b",
        "name": "gemma_7b_lora"
    },
    # Comment out models you don't need
    # {
    #     "base_model": "tiiuae/falcon-7b-instruct",
    #     "lora_adapter": "./lora-models/falcon-7b",
    #     "name": "falcon_7b_lora"
    # },
]
```

#### Troubleshooting

**"Question column not found"**: Ensure your CSV has a column named exactly `Question` (case-sensitive).

**Out of memory errors**: Reduce `MAX_NEW_TOKENS` or process fewer models at once.

**Model loading errors**: Check that your LoRA adapter directory contains:
- `adapter_config.json`
- `adapter_model.safetensors` (or `adapter_model.bin`)

**Empty responses**: Try increasing `TEMPERATURE` for more varied outputs, or check if the model loaded correctly.

## Using Pre-trained LoRA Adapters

### Load and Use a LoRA Adapter

Here's how to load and use any of the pre-trained LoRA adapters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Choose your model and adapter
base_model_name = "tiiuae/falcon-7b-instruct"
lora_adapter = "KraveTech/Falcon7B_LoRA"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, lora_adapter)

# Run inference
prompt = "What is artificial intelligence?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Example for Each Model

#### Gemma 7B
```python
base_model = "google/gemma-7b-it"
lora_adapter = "KraveTech/Gemma7B_LoRA"
```

#### Llama 2 7B
```python
base_model = "meta-llama/Llama-2-7b-chat-hf"
lora_adapter = "KraveTech/Llama7B_LoRA"
```

#### Qwen 2.5 7B
```python
base_model = "Qwen/Qwen2.5-7B-Instruct"
lora_adapter = "KraveTech/Qwen7B_LoRA"
```

#### Mistral 7B
```python
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_adapter = "KraveTech/Mistral7B_LoRA"
```

#### Granite 7B
```python
base_model = "ibm-granite/granite-7b-instruct"
lora_adapter = "KraveTech/Granite7B_LoRA"
```

## Project Structure

```
.
├── train_models.py           # Training script for all models
├── inference_lora_csv.py     # Batch inference from CSV
├── README.md                 # This file
└── data/
│   └── train.jsonl           # Training data (JSONL format)
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- Hugging Face account (for accessing gated models like Llama)

## Notes

- Some models (like Llama 2) require accepting their license on Hugging Face before use
- Training time varies by model and dataset size
- For production use, consider quantization (4-bit/8-bit) to reduce memory requirements

## License

Please refer to the individual model licenses on Hugging Face for usage terms.

## Citation

If you use these LoRA adapters, please cite the original model papers and the PEFT library:

```bibtex
@article{tham2025lora7b,
  title={Understanding LoRA Fine-Tuning Behavior Across Heterogeneous 7B LLM Architectures},
  author={Tham Hiu Huen, Kathleen Tan Swee Neo, Lim Tong Ming},
  journal={To be confirm},
  year={2025}
}
```
