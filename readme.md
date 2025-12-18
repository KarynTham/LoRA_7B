# LoRA Fine-tuning for 7B Instruction Models

This repository contains scripts for training and inference using LoRA (Low-Rank Adaptation) on six popular 7B instruction-tuned language models.

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

The training data should be in **JSONL format** (JSON Lines), where each line is a separate JSON object with the following structure:

```json
{"instruction": "Example Question?", "input": "", "output": "Example Response 1"}
{"instruction": "What is machine learning?", "input": "", "output": "Machine learning is a subset of artificial intelligence..."}
{"instruction": "Explain neural networks.", "input": "", "output": "Neural networks are computational models inspired by the human brain..."}
```

**Field Descriptions:**
- `instruction`: The question or task prompt
- `input`: Additional context (can be empty string if not needed)
- `output`: The expected response or answer

Save your training data as `train.jsonl` in the `data/` directory.

### Train All Models

To train LoRA adapters for all six models:

```bash
python train_models.py
```

This script will:
- Load training data from `data/train.jsonl`
- Load each base model
- Apply LoRA configuration
- Fine-tune on your training dataset
- Save the LoRA adapters to local directories

### Training Configuration

You can modify training parameters in `train_models.py` such as:
- Learning rate
- Batch size
- Number of epochs
- LoRA rank and alpha
- Target modules

## Inference

### Batch Inference from CSV

To run inference on questions from a CSV file and save responses:

```bash
python inference_lora_csv.py
```

**Input**: CSV file containing questions (e.g., `questions.csv`)
**Output**: CSV file with model responses (e.g., `responses.csv`)

The script will:
1. Load questions from the input CSV
2. Process each question through the LoRA-adapted model
3. Save responses to an output CSV file

### CSV Format

**Input CSV** should have at least a column with questions:
```csv
question
What is machine learning?
Explain neural networks.
```

**Output CSV** will include both questions and responses:
```csv
question,response
What is machine learning?,"Machine learning is..."
Explain neural networks.,"Neural networks are..."
```

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
├── data/
    └── train.jsonl          # Training data (JSONL format)
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
@article{hu2021lora,
  title={Understanding LoRA Fine-Tuning Behavior Across Heterogeneous 7B LLM Architectures},
  author={Tham Hiu Huen, Kathleen Tan Swee Neo, Lim Tong Ming},
  journal={To be confirm},
  year={2025}
}
```
