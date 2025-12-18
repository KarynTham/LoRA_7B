import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import pandas as pd
import warnings
from tqdm import tqdm
import os
import traceback
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

# Model configurations with base model and LoRA adapter paths
MODELS = [
    {
        "base_model": "google/gemma-7b-it",
        "lora_adapter": "./lora-models/gemma-7b",
        "name": "gemma_7b_lora"
    },
    # {
    #     "base_model": "tiiuae/falcon-7b-instruct",
    #     "lora_adapter": "./lora-models/falcon-7b",
    #     "name": "falcon_7b_lora"
    # },
    # {
    #     "base_model": "meta-llama/Llama-2-7b-chat-hf",
    #     "lora_adapter": "./lora-models/llama2-7b",
    #     "name": "llama2_7b_lora"
    # },
    # {
    #     "base_model": "Qwen/Qwen2.5-7B-Instruct",
    #     "lora_adapter": "./lora-models/qwen-7b",
    #     "name": "qwen_7b_lora"
    # },
    # {
    #     "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
    #     "lora_adapter": "./lora-models/mistral-7b",
    #     "name": "mistral_7b_lora"
    # },
    # {
    #     "base_model": "ibm-granite/granite-7b-instruct",
    #     "lora_adapter": "./lora-models/granite-7b",
    #     "name": "granite_7b_lora"
    # },
]


def load_lora_model(base_model_name, lora_adapter_path):
    """
    Load base model with LoRA adapter
    
    Args:
        base_model_name: HuggingFace base model identifier
        lora_adapter_path: Path to LoRA adapter (local or HF hub)
    
    Returns:
        Tuple of (tokenizer, model) or (None, None) on failure
    """
    print(f"\nLoading base model: {base_model_name}...")
    print(f"Loading LoRA adapter: {lora_adapter_path}...")
    
    # Check if adapter_config.json exists
    adapter_config_path = os.path.join(lora_adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"\n⚠️  WARNING: adapter_config.json not found at {lora_adapter_path}")
        print(f"Available files in directory:")
        if os.path.exists(lora_adapter_path):
            files = os.listdir(lora_adapter_path)
            for f in files:
                print(f"  - {f}")
        else:
            print(f"  Directory does not exist!")
        print("\nPlease ensure your LoRA directory contains:")
        print("  - adapter_config.json")
        print("  - adapter_model.safetensors (or adapter_model.bin)")
        return None, None
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Special handling for Falcon models
        is_falcon = "falcon" in base_model_name.lower()
        
        # Load tokenizer from base model
        if is_falcon:
            print("Loading Falcon tokenizer with special configuration...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            # Falcon needs explicit pad token setup
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, 
                trust_remote_code=True
            )
            # Set pad token if not defined
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Handle padding side for decoder-only models
        tokenizer.padding_side = "left"
        
        # Set model_max_length to avoid truncation warnings
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 100000:
            tokenizer.model_max_length = 2048
        
        # Load base model with proper device mapping
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter and merge with base model
        model = PeftModel.from_pretrained(
            base_model, 
            lora_adapter_path,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Ensure model is in eval mode
        model.eval()
        
        # Optional: Merge LoRA weights into base model for faster inference
        # Uncomment the line below if you want to merge (uses more VRAM but faster)
        # model = model.merge_and_unload()
        
        # Verify tokenizer is working
        print("Testing tokenizer...")
        test_encoding = tokenizer("test", return_tensors="pt")
        if test_encoding is None or "input_ids" not in test_encoding:
            raise ValueError("Tokenizer test failed - unable to encode text")
        print(f"  Test encoding shape: {test_encoding['input_ids'].shape}")
        
        print(f"✓ Model with LoRA adapter loaded successfully on {device}!")
        print(f"  Tokenizer vocab size: {len(tokenizer)}")
        print(f"  Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        return tokenizer, model
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        traceback.print_exc()
        return None, None


def load_lora_model_auto(lora_adapter_path):
    """
    Load model using LoRA config (automatically detects base model)
    
    Args:
        lora_adapter_path: Path to LoRA adapter (local or HF hub)
    
    Returns:
        Tuple of (tokenizer, model) or (None, None) on failure
    """
    print(f"\nLoading LoRA model from: {lora_adapter_path}...")
    
    try:
        # Load LoRA config to get base model info
        config = PeftConfig.from_pretrained(lora_adapter_path)
        base_model_name = config.base_model_name_or_path
        
        print(f"Detected base model: {base_model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model, 
            lora_adapter_path,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Ensure model is in eval mode
        model.eval()
        
        # Set pad token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        tokenizer.padding_side = "left"
        
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 100000:
            tokenizer.model_max_length = 2048
        
        print(f"✓ Model with LoRA adapter loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        traceback.print_exc()
        return None, None


def strip_prompt_from_response(response, prompt):
    """
    Remove the original prompt from the model's response
    
    Args:
        response: Full model output
        prompt: Original input prompt
    
    Returns:
        Response with prompt removed
    """
    # Try exact match removal
    if response.startswith(prompt):
        return response[len(prompt):].strip()
    
    # Try case-insensitive match
    if response.lower().startswith(prompt.lower()):
        return response[len(prompt):].strip()
    
    # Try to find prompt with minor variations (extra spaces, newlines)
    prompt_normalized = " ".join(prompt.split())
    response_normalized = " ".join(response.split())
    
    if response_normalized.startswith(prompt_normalized):
        # Find where the prompt ends in the original response
        words = prompt.split()
        for i in range(len(response)):
            candidate = response[i:]
            if not any(word in candidate[:100] for word in words[-3:] if len(word) > 3):
                return candidate.strip()
    
    # Return original if no match found
    return response.strip()


def inference_single(tokenizer, model, prompt, max_new_tokens=256, temperature=0.7):
    """
    Run inference with pre-loaded LoRA model
    
    Args:
        tokenizer: Pre-loaded tokenizer
        model: Pre-loaded model with LoRA
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text output (without the original prompt)
    """
    try:
        # Validate inputs
        if tokenizer is None or model is None:
            return "Error: Tokenizer or model is None"
        
        if not prompt or str(prompt).strip() == "":
            return "Error: Empty prompt provided"
        
        # Get model device
        device = next(model.parameters()).device
        
        # Check if this is a Falcon model
        model_name = model.config._name_or_path if hasattr(model.config, '_name_or_path') else ""
        is_falcon = "falcon" in model_name.lower()
        
        # Tokenize input
        inputs = tokenizer(
            str(prompt), 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        
        # Check if tokenization succeeded
        if "input_ids" not in inputs or inputs["input_ids"] is None:
            return "Error: Tokenization failed - input_ids is None"
        
        # Store input length for later trimming
        input_length = inputs["input_ids"].shape[1]
        
        # Move all inputs to the same device as model
        inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
        
        # Verify input_ids is valid
        if inputs["input_ids"] is None or inputs["input_ids"].shape[0] == 0:
            return "Error: Invalid input_ids after tokenization"
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 0 else False,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Falcon-specific fixes for the cache issue
        if is_falcon:
            gen_kwargs["use_cache"] = False  # Disable KV cache to avoid the NoneType error
            gen_kwargs["return_dict_in_generate"] = False
        
        # Only add temperature and top_p if doing sampling
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        
        # Add attention_mask if available
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            gen_kwargs["attention_mask"] = inputs["attention_mask"]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                **gen_kwargs
            )
        
        # Decode only the newly generated tokens (exclude input prompt)
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Fallback: if the above doesn't work, try stripping the prompt manually
        if not response.strip():
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = strip_prompt_from_response(full_response, str(prompt))
        
        return response.strip()
        
    except Exception as e:
        error_details = traceback.format_exc()
        return f"Error: {str(e)}\nDetails: {error_details}"


def process_csv_with_lora_models(csv_path, output_path=None, max_new_tokens=256, 
                                  temperature=0.7, save_interval=10):
    """
    Process all questions in CSV with all LoRA models
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to save output CSV
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        save_interval: Save progress every N rows
    
    Returns:
        DataFrame with results
    """
    # Read CSV
    print(f"Reading CSV from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'Question' not in df.columns:
        raise ValueError(f"'Question' column not found. Available columns: {df.columns.tolist()}")
    
    print(f"Found {len(df)} rows with questions")
    
    # Create columns for each model
    for model_config in MODELS:
        column_name = model_config["name"]
        if column_name not in df.columns:
            df[column_name] = ""
    
    # Set output path
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_lora_results.csv"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Output will be saved to: {output_path}")
    print("="*80)
    
    # Process each model
    for model_idx, model_config in enumerate(MODELS, 1):
        column_name = model_config["name"]
        base_model = model_config["base_model"]
        lora_adapter = model_config["lora_adapter"]
        
        print(f"\n{'='*80}")
        print(f"Processing model {model_idx}/{len(MODELS)}: {column_name}")
        print(f"Base model: {base_model}")
        print(f"LoRA adapter: {lora_adapter}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Load model with LoRA adapter
        tokenizer, model = load_lora_model(base_model, lora_adapter)
        
        # Alternative: Use auto-detection
        # tokenizer, model = load_lora_model_auto(lora_adapter)
        
        if tokenizer is None or model is None:
            print(f"⚠️  Skipping {column_name} due to loading error\n")
            df[column_name] = df[column_name].fillna("Error: Model failed to load")
            continue
        
        # Count how many rows need processing
        rows_to_process = df[column_name].isna() | (df[column_name] == "")
        num_to_process = rows_to_process.sum()
        print(f"Rows to process: {num_to_process}/{len(df)}")
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {column_name}"):
            # Skip if already processed
            if pd.notna(df.at[idx, column_name]) and df.at[idx, column_name] != "":
                continue
            
            question = row['Question']
            
            if pd.isna(question) or str(question).strip() == "":
                df.at[idx, column_name] = "No question provided"
                continue
            
            # Run inference
            response = inference_single(
                tokenizer, 
                model, 
                str(question), 
                max_new_tokens, 
                temperature
            )
            
            # Save response
            df.at[idx, column_name] = response
            
            # Save progress at intervals
            if (idx + 1) % save_interval == 0:
                df.to_csv(output_path, index=False)
                print(f"\n💾 Progress saved at row {idx + 1}/{len(df)}")
        
        # Clean up memory
        print(f"\n🧹 Cleaning up memory for {column_name}...")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save after completing each model
        df.to_csv(output_path, index=False)
        print(f"\n✓ Completed {column_name}. Results saved to {output_path}")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("🎉 ALL MODELS COMPLETED!")
    print(f"Final results saved to: {output_path}")
    print(f"Total rows processed: {len(df)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return df


if __name__ == "__main__":
    # Configuration
    CSV_PATH = "QA_Evaluation.csv"
    OUTPUT_PATH = "QA_Evaluation_Lora.csv"
    
    # Generation parameters
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.5
    SAVE_INTERVAL = 10
    
    print("="*80)
    print("LoRA Model Batch Inference Script")
    print("="*80)
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output CSV: {OUTPUT_PATH}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Save interval: {SAVE_INTERVAL} rows")
    print(f"Number of models: {len(MODELS)}")
    print("="*80)
    
    # Process the CSV
    try:
        results_df = process_csv_with_lora_models(
            csv_path=CSV_PATH,
            output_path=OUTPUT_PATH,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            save_interval=SAVE_INTERVAL
        )
        
        print(f"\n✅ Processing complete! Check {OUTPUT_PATH} for results.")
        print(f"Total rows: {len(results_df)}")
        print(f"Output columns: {results_df.columns.tolist()}")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        traceback.print_exc()