import os
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
import warnings
import gc
warnings.filterwarnings('ignore')

# ============================================================
# Model Configurations
# ============================================================

MODELS = {
    "tiiuae/falcon-7b-instruct": {
        "name": "falcon-7b",
        "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    },    
    "google/gemma-7b-it": {
        "name": "gemma-7b",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "name": "llama2-7b",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "name": "qwen-7b",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "name": "mistral-7b",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "ibm-granite/granite-7b-instruct": {
        "name": "granite-7b",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
}

# ============================================================
# Shared Training Configuration
# ============================================================

CONFIG = {
    "output_base_dir": "./lora-models",  # Base directory for all models
    "dataset_name": None, # "timdettmers/openassistant-guanaco",  # HuggingFace dataset
    "dataset_file": "data/train.jsonl",  # Path to local JSONL/JSON file (e.g., "train.jsonl")
    "max_seq_length": 1024,
    
    # Training mode: "all", "sequential", or list of model names to train
    # Examples:
    # "train_mode": "all"  # Train all models at once
    # "train_mode": ["gemma-7b", "llama2-7b"]  # Train only specific models
    # "train_mode": "sequential"  # Same as "all" but more explicit
    "train_mode": ["gemma-7b", "mistral-7b"],
    
    # Skip models that already have training_metrics.json (useful for resume)
    "skip_completed": False,
    
    # Quantization parameters (QLoRA with 4-bit)
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "use_nested_quant": False,
    
    # Memory optimization
    "max_memory": None,  # e.g., {"0": "10GB", "cpu": "30GB"} for custom allocation
    "low_cpu_mem_usage": True,
    "llm_int8_enable_fp32_cpu_offload": True,  # Enable CPU offload for tight memory
    
    # LoRA parameters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    
    # Training parameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,  # Reduce memory usage
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 3,
    "fp16": True,
    "optim": "paged_adamw_8bit",
}

# ============================================================
# Metrics Tracking
# ============================================================

class MetricsCallback(TrainerCallback):
    """Custom callback to track detailed training metrics"""
    
    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rates": [],
            "gradient_norms": [],
            "step_times": [],
            "gpu_memory": [],
            "nan_detected": False,
            "divergence_detected": False,
            "steps": []
        }
        self.start_time = None
        self.step_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        # Record step time
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            self.metrics["step_times"].append(step_time)
        
        # Record GPU memory
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB
            self.metrics["gpu_memory"].append(memory_used)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.metrics["steps"].append(state.global_step)
            
            # Track losses
            if "loss" in logs:
                self.metrics["train_loss"].append(logs["loss"])
                
                # Check for NaN or divergence
                if np.isnan(logs["loss"]) or np.isinf(logs["loss"]):
                    self.metrics["nan_detected"] = True
                if logs["loss"] > 100:
                    self.metrics["divergence_detected"] = True
            
            if "eval_loss" in logs:
                self.metrics["eval_loss"].append(logs["eval_loss"])
            
            # Track learning rate
            if "learning_rate" in logs:
                self.metrics["learning_rates"].append(logs["learning_rate"])
            
            # Track gradient norm
            if "grad_norm" in logs:
                self.metrics["gradient_norms"].append(logs["grad_norm"])

class TrainingMetrics:
    """Container for all training metrics and analysis"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = None
        self.end_time = None
        self.callback_metrics = None
        self.lora_params = {}
        self.model_config = {}
        self.dataset_stats = {}
        self.final_metrics = {}
        
    def calculate_convergence(self, losses, window=10):
        """Calculate convergence metrics"""
        if len(losses) < window * 2:
            return {"converged": False, "reason": "insufficient_data"}
        
        recent_losses = losses[-window:]
        previous_losses = losses[-window*2:-window]
        
        recent_mean = np.mean(recent_losses)
        previous_mean = np.mean(previous_losses)
        recent_std = np.std(recent_losses)
        
        improvement = (previous_mean - recent_mean) / previous_mean if previous_mean > 0 else 0
        
        converged = improvement < 0.01 and recent_std < 0.1
        
        return {
            "converged": converged,
            "improvement_rate": improvement,
            "recent_std": recent_std,
            "recent_mean": recent_mean
        }
    
    def calculate_gradient_stability(self, grad_norms):
        """Analyze gradient stability"""
        if len(grad_norms) == 0:
            return {}
        
        grad_norms = np.array(grad_norms)
        
        return {
            "mean": float(np.mean(grad_norms)),
            "std": float(np.std(grad_norms)),
            "max": float(np.max(grad_norms)),
            "min": float(np.min(grad_norms)),
            "stability_score": float(np.std(grad_norms) / (np.mean(grad_norms) + 1e-8))
        }
    
    def calculate_train_val_gap(self, train_losses, eval_losses):
        """Calculate train-validation loss gap"""
        if len(train_losses) == 0 or len(eval_losses) == 0:
            return {}
        
        # Use final losses
        final_train = train_losses[-1] if train_losses else 0
        final_eval = eval_losses[-1] if eval_losses else 0
        
        gap = abs(final_eval - final_train)
        relative_gap = gap / final_train if final_train > 0 else 0
        
        overfitting = final_eval > final_train * 1.1
        
        return {
            "absolute_gap": float(gap),
            "relative_gap": float(relative_gap),
            "final_train_loss": float(final_train),
            "final_eval_loss": float(final_eval),
            "overfitting": overfitting
        }
    
    def analyze_lora_updates(self, model):
        """Analyze LoRA adapter magnitude"""
        lora_magnitudes = []
        
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                magnitude = torch.norm(param.data).item()
                lora_magnitudes.append({
                    "layer": name,
                    "magnitude": magnitude,
                    "shape": list(param.shape)
                })
        
        if lora_magnitudes:
            mags = [m["magnitude"] for m in lora_magnitudes]
            return {
                "layers": lora_magnitudes,
                "mean_magnitude": float(np.mean(mags)),
                "max_magnitude": float(np.max(mags)),
                "min_magnitude": float(np.min(mags))
            }
        return {}
    
    def compile_report(self):
        """Compile final metrics report"""
        if not self.callback_metrics:
            return {}
        
        training_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate tokens per second
        total_steps = len(self.callback_metrics.metrics["steps"])
        tokens_per_step = self.dataset_stats.get("batch_size", 1) * self.dataset_stats.get("seq_length", 1)
        total_tokens = total_steps * tokens_per_step
        tokens_per_second = total_tokens / training_time if training_time > 0 else 0
        
        # Helper function to convert numpy/torch types to native Python types
        def convert_to_native(obj):
            """Convert numpy/torch types to native Python types for JSON serialization"""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        report = {
            "model_name": self.model_name,
            "training_time_seconds": float(training_time),
            "training_time_formatted": f"{training_time/3600:.2f}h" if training_time > 3600 else f"{training_time/60:.2f}m",
            
            # Loss metrics
            "loss_curve": {
                "train_losses": [float(x) for x in self.callback_metrics.metrics["train_loss"]],
                "eval_losses": [float(x) for x in self.callback_metrics.metrics["eval_loss"]],
                "steps": [int(x) for x in self.callback_metrics.metrics["steps"]]
            },
            
            # Convergence
            "convergence": convert_to_native(self.calculate_convergence(self.callback_metrics.metrics["train_loss"])),
            
            # Gradient stability
            "gradient_stability": convert_to_native(self.calculate_gradient_stability(self.callback_metrics.metrics["gradient_norms"])),
            
            # Train-val gap
            "train_val_gap": convert_to_native(self.calculate_train_val_gap(
                self.callback_metrics.metrics["train_loss"],
                self.callback_metrics.metrics["eval_loss"]
            )),
            
            # LoRA metrics
            "lora_config": convert_to_native(self.lora_params),
            "lora_updates": convert_to_native(self.final_metrics.get("lora_updates", {})),
            
            # GPU metrics
            "gpu_vram": {
                "peak_gb": float(np.max(self.callback_metrics.metrics["gpu_memory"])) if self.callback_metrics.metrics["gpu_memory"] else 0.0,
                "mean_gb": float(np.mean(self.callback_metrics.metrics["gpu_memory"])) if self.callback_metrics.metrics["gpu_memory"] else 0.0
            },
            
            # Throughput
            "throughput": {
                "tokens_per_second": float(tokens_per_second),
                "samples_per_second": float(total_steps / training_time if training_time > 0 else 0),
                "mean_step_time": float(np.mean(self.callback_metrics.metrics["step_times"])) if self.callback_metrics.metrics["step_times"] else 0.0
            },
            
            # Stability indicators
            "stability": {
                "nan_detected": bool(self.callback_metrics.metrics["nan_detected"]),
                "divergence_detected": bool(self.callback_metrics.metrics["divergence_detected"]),
                "training_stable": bool(not (self.callback_metrics.metrics["nan_detected"] or self.callback_metrics.metrics["divergence_detected"]))
            },
            
            # Dataset info
            "dataset_stats": convert_to_native(self.dataset_stats),
            
            # Model config
            "model_config": convert_to_native(self.model_config)
        }
        
        return report
    
    def save_report(self, output_dir):
        """Save report to JSON and generate plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = self.compile_report()
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Save JSON report
        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        # Generate plots
        self._plot_loss_curves(output_dir, report)
        self._plot_gradient_norms(output_dir, report)
        self._plot_learning_rate(output_dir, report)
        self._plot_gpu_memory(output_dir, report)
        
        # Generate text report
        self._generate_text_report(output_dir, report)
        
        return report
    
    def _plot_loss_curves(self, output_dir, report):
        """Plot training and validation loss curves"""
        try:
            plt.figure(figsize=(10, 6))
            
            steps = report["loss_curve"]["steps"]
            train_losses = report["loss_curve"]["train_losses"]
            
            if train_losses:
                plt.plot(steps[:len(train_losses)], train_losses, label="Train Loss", linewidth=2)
            
            if report["loss_curve"]["eval_losses"]:
                # Eval losses are sparse, need to map to steps
                eval_steps = steps[::len(steps)//len(report["loss_curve"]["eval_losses"])] if report["loss_curve"]["eval_losses"] else []
                eval_steps = eval_steps[:len(report["loss_curve"]["eval_losses"])]
                plt.plot(eval_steps, report["loss_curve"]["eval_losses"], 
                        label="Eval Loss", linewidth=2, linestyle="--", marker="o")
            
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title(f"Loss Curves - {self.model_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot loss curves: {e}")
    
    def _plot_gradient_norms(self, output_dir, report):
        """Plot gradient norms over time"""
        try:
            if not self.callback_metrics.metrics["gradient_norms"]:
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.callback_metrics.metrics["gradient_norms"], linewidth=1, alpha=0.7)
            plt.xlabel("Steps")
            plt.ylabel("Gradient Norm")
            plt.title(f"Gradient Norms - {self.model_name}")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "gradient_norms.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot gradient norms: {e}")
    
    def _plot_learning_rate(self, output_dir, report):
        """Plot learning rate schedule"""
        try:
            if not self.callback_metrics.metrics["learning_rates"]:
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.callback_metrics.metrics["learning_rates"], linewidth=2)
            plt.xlabel("Steps")
            plt.ylabel("Learning Rate")
            plt.title(f"Learning Rate Schedule - {self.model_name}")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot learning rate: {e}")
    
    def _plot_gpu_memory(self, output_dir, report):
        """Plot GPU memory usage"""
        try:
            if not self.callback_metrics.metrics["gpu_memory"]:
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.callback_metrics.metrics["gpu_memory"], linewidth=1, alpha=0.7)
            plt.xlabel("Steps")
            plt.ylabel("GPU Memory (GB)")
            plt.title(f"GPU Memory Usage - {self.model_name}")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "gpu_memory.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot GPU memory: {e}")
    
    def _generate_text_report(self, output_dir, report):
        """Generate human-readable text report"""
        with open(os.path.join(output_dir, "training_report.txt"), "w") as f:
            f.write("="*70 + "\n")
            f.write(f"TRAINING REPORT: {self.model_name}\n")
            f.write("="*70 + "\n\n")
            
            # Model information
            f.write("MODEL INFORMATION\n")
            f.write("-"*70 + "\n")
            model_cfg = report.get('model_config', {})
            f.write(f"Model Path: {model_cfg.get('model_path', 'N/A')}\n")
            f.write(f"Model Name: {model_cfg.get('model_name', 'N/A')}\n")
            f.write(f"Quantization: {model_cfg.get('quantization', 'N/A')}\n")
            
            vocab_size = model_cfg.get('vocab_size')
            tokenizer_vocab = model_cfg.get('tokenizer_vocab_size')
            if vocab_size and tokenizer_vocab:
                f.write(f"Model Vocab Size: {vocab_size:,}\n")
                f.write(f"Tokenizer Vocab Size: {tokenizer_vocab:,}\n")
                if vocab_size != tokenizer_vocab:
                    f.write(f"  ⚠ Mismatch: {abs(vocab_size - tokenizer_vocab)} tokens difference\n")
            
            if model_cfg.get('hidden_size'):
                f.write(f"Hidden Size: {model_cfg['hidden_size']}\n")
            if model_cfg.get('num_layers'):
                f.write(f"Number of Layers: {model_cfg['num_layers']}\n")
            if model_cfg.get('num_attention_heads'):
                f.write(f"Attention Heads: {model_cfg['num_attention_heads']}\n")
            
            # Trainable parameters
            f.write("\nTRAINABLE PARAMETERS\n")
            f.write("-"*70 + "\n")
            trainable = model_cfg.get('trainable_params', 0)
            total = model_cfg.get('all_params', 0)
            percent = model_cfg.get('trainable_percent', 0)
            
            f.write(f"Trainable params: {trainable:,}\n")
            f.write(f"All params: {total:,}\n")
            f.write(f"Trainable %: {percent:.4f}%\n")
            f.write(f"Frozen params: {total - trainable:,}\n\n")
            
            # Training overview
            f.write("TRAINING OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"Training Time: {report['training_time_formatted']}\n")
            f.write(f"Total Steps: {len(report['loss_curve']['steps'])}\n")
            f.write(f"Tokens/Second: {report['throughput']['tokens_per_second']:.2f}\n")
            f.write(f"Samples/Second: {report['throughput']['samples_per_second']:.2f}\n\n")
            
            # Loss metrics
            f.write("LOSS METRICS\n")
            f.write("-"*70 + "\n")
            if report['loss_curve']['train_losses']:
                f.write(f"Final Train Loss: {report['loss_curve']['train_losses'][-1]:.4f}\n")
            if report['loss_curve']['eval_losses']:
                f.write(f"Final Eval Loss: {report['loss_curve']['eval_losses'][-1]:.4f}\n")
            
            gap = report['train_val_gap']
            if gap:
                f.write(f"Train-Val Gap: {gap['absolute_gap']:.4f} ({gap['relative_gap']*100:.2f}%)\n")
                f.write(f"Overfitting: {'Yes' if gap['overfitting'] else 'No'}\n\n")
            
            # Convergence
            f.write("CONVERGENCE\n")
            f.write("-"*70 + "\n")
            conv = report['convergence']
            f.write(f"Converged: {'Yes' if conv.get('converged') else 'No'}\n")
            if 'improvement_rate' in conv:
                f.write(f"Recent Improvement Rate: {conv['improvement_rate']*100:.2f}%\n")
            if 'recent_std' in conv:
                f.write(f"Recent Loss Std: {conv['recent_std']:.4f}\n\n")
            
            # Gradient stability
            f.write("GRADIENT STABILITY\n")
            f.write("-"*70 + "\n")
            grad = report['gradient_stability']
            if grad:
                f.write(f"Mean Gradient Norm: {grad['mean']:.4f}\n")
                f.write(f"Std Gradient Norm: {grad['std']:.4f}\n")
                f.write(f"Stability Score: {grad['stability_score']:.4f}\n\n")
            
            # LoRA metrics
            f.write("LORA CONFIGURATION\n")
            f.write("-"*70 + "\n")
            lora = report['lora_config']
            f.write(f"Rank (r): {lora.get('r', 'N/A')}\n")
            f.write(f"Alpha: {lora.get('alpha', 'N/A')}\n")
            f.write(f"Dropout: {lora.get('dropout', 'N/A')}\n")
            f.write(f"Target Modules: {', '.join(lora.get('target_modules', []))}\n")
            
            # LoRA update magnitudes
            lora_updates = report.get('lora_updates', {})
            if lora_updates and 'mean_magnitude' in lora_updates:
                f.write(f"\nLoRA Update Statistics:\n")
                f.write(f"  Mean Magnitude: {lora_updates['mean_magnitude']:.6f}\n")
                f.write(f"  Max Magnitude: {lora_updates['max_magnitude']:.6f}\n")
                f.write(f"  Min Magnitude: {lora_updates['min_magnitude']:.6f}\n")
            f.write("\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            dataset = report['dataset_stats']
            f.write(f"Train Size: {dataset.get('train_size', 'N/A')}\n")
            f.write(f"Eval Size: {dataset.get('eval_size', 'N/A')}\n")
            f.write(f"Batch Size (effective): {dataset.get('batch_size', 'N/A')}\n")
            f.write(f"Sequence Length: {dataset.get('seq_length', 'N/A')}\n\n")
            
            # GPU metrics
            f.write("GPU METRICS\n")
            f.write("-"*70 + "\n")
            gpu = report['gpu_vram']
            f.write(f"Peak VRAM: {gpu['peak_gb']:.2f} GB\n")
            f.write(f"Mean VRAM: {gpu['mean_gb']:.2f} GB\n\n")
            
            # Stability
            f.write("STABILITY\n")
            f.write("-"*70 + "\n")
            stab = report['stability']
            f.write(f"NaN Detected: {'Yes' if stab['nan_detected'] else 'No'}\n")
            f.write(f"Divergence Detected: {'Yes' if stab['divergence_detected'] else 'No'}\n")
            f.write(f"Training Stable: {'Yes' if stab['training_stable'] else 'No'}\n\n")
            
            f.write("="*70 + "\n")

# ============================================================
# Load and Prepare Model
# ============================================================

def load_model_and_tokenizer(model_name, config):
    """Load model and tokenizer with 4-bit quantization"""
    
    print(f"\n{'='*70}")
    print(f"Loading: {model_name}")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # CRITICAL: Ensure model_max_length is set properly
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e6:
        tokenizer.model_max_length = config["max_seq_length"]
        print(f"✓ Set tokenizer.model_max_length to {config['max_seq_length']}")
    
    print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
    print(f"✓ Tokenizer max length: {tokenizer.model_max_length}")
    
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Available GPU memory: {gpu_memory:.2f} GB")
        
        # Auto-configure max_memory if not set
        if config.get("max_memory") is None:
            # Reserve some memory for operations
            available_gpu = int(gpu_memory * 0.85)  # Use 85% of GPU
            config["max_memory"] = {0: f"{available_gpu}GB", "cpu": "30GB"}
            print(f"✓ Auto-configured max_memory: {config['max_memory']}")
    
    # Configure 4-bit quantization
    if config["use_4bit"]:
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["use_4bit"],
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config["use_nested_quant"],
            llm_int8_enable_fp32_cpu_offload=config.get("llm_int8_enable_fp32_cpu_offload", True),
        )
        
        print(f"✓ Using 4-bit quantization (QLoRA)")
        print(f"  - CPU offload: {config.get('llm_int8_enable_fp32_cpu_offload', True)}")
    else:
        bnb_config = None
        print(f"✓ Using FP16 (no quantization)")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=config.get("max_memory"),
            low_cpu_mem_usage=config.get("low_cpu_mem_usage", True),
            trust_remote_code=True
        )
    except ValueError as e:
        if "CPU or the disk" in str(e):
            print(f"⚠ GPU memory insufficient, enabling aggressive CPU offloading...")
            # Retry with more aggressive settings
            bnb_config.llm_int8_enable_fp32_cpu_offload = True
            
            # Create custom device map with more CPU offloading
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                max_memory = {0: f"{int(gpu_memory_gb * 0.7)}GB", "cpu": "40GB"}
            else:
                max_memory = None
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="offload",
                offload_state_dict=True
            )
            print(f"✓ Model loaded with CPU offloading")
        else:
            raise
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Ensure model vocab size matches tokenizer
    model_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = len(tokenizer)
    
    if model_vocab_size != tokenizer_vocab_size:
        print(f"⚠ WARNING: Vocab size mismatch!")
        print(f"  Model vocab size: {model_vocab_size}")
        print(f"  Tokenizer vocab size: {tokenizer_vocab_size}")
        
        # Resize model embeddings to match tokenizer
        if model_vocab_size > tokenizer_vocab_size:
            print(f"  Model has extra embeddings, this is usually fine")
            # Some models have reserved tokens, this is normal
        else:
            print(f"  Resizing model embeddings to match tokenizer...")
            model.resize_token_embeddings(tokenizer_vocab_size)
    
    return model, tokenizer

def setup_lora(model, target_modules, config):
    """Configure and apply LoRA to the model"""
    
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing for memory savings
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print(f"✓ Gradient checkpointing enabled")
    
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    print(f"✓ Applying LoRA configuration...")
    print(f"  Target modules: {target_modules}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# ============================================================
# Dataset Preparation
# ============================================================

def prepare_dataset(dataset_config, tokenizer, max_seq_length):
    """Load and prepare dataset for training"""
    
    # Load dataset from file or HuggingFace
    if dataset_config.get("dataset_file"):
        print(f"✓ Loading dataset from file: {dataset_config['dataset_file']}...")
        
        # Detect file type
        if dataset_config["dataset_file"].endswith(".jsonl"):
            dataset = load_dataset("json", data_files=dataset_config["dataset_file"], split="train")
        elif dataset_config["dataset_file"].endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_config["dataset_file"], split="train")
        elif dataset_config["dataset_file"].endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_config["dataset_file"], split="train")
        else:
            raise ValueError(f"Unsupported file format: {dataset_config['dataset_file']}")
    else:
        print(f"✓ Loading dataset from HuggingFace: {dataset_config['dataset_name']}...")
        dataset = load_dataset(dataset_config["dataset_name"], split="train")
    
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    
    def format_instruction(example):
        """Format the dataset into instruction-response pairs"""
        # Handle different dataset formats
        if "text" in example and example["text"]:
            # Already formatted text
            return {"text": example["text"]}
        elif "instruction" in example and "output" in example:
            # Standard instruction-output format
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]
            
            # Format with optional input field
            if input_text and input_text.strip():
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            return {"text": text}
        elif "prompt" in example and "completion" in example:
            # Prompt-completion format
            text = f"{example['prompt']}\n{example['completion']}"
            return {"text": text}
        else:
            # Try to use the example as-is
            return example
    
    print(f"✓ Formatting dataset...")
    dataset = dataset.map(format_instruction)
    
    # Get vocab size for validation
    # vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else len(tokenizer)
    vocab_size = len(tokenizer)
    print(f"✓ Using vocab size for validation: {vocab_size}")
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        
        # CRITICAL: Validate token IDs are within vocab size
        # This prevents CUDA indexing errors
        for i, input_ids in enumerate(result["input_ids"]):
            # Check for out-of-bounds token IDs
            max_id = max(input_ids)
            if max_id >= vocab_size:
                print(f"⚠ Warning: Found token ID {max_id} >= vocab_size {vocab_size} in example {i}")
                # Clip invalid token IDs to vocab_size - 1
                result["input_ids"][i] = [min(tid, vocab_size - 1) for tid in input_ids]
        
        result["labels"] = result["input_ids"].copy()
        return result
    
    print(f"✓ Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Validate tokenized dataset
    print(f"✓ Validating tokenized dataset...")
    sample = tokenized_dataset[0]
    max_token_id = max(sample["input_ids"])
    if max_token_id >= vocab_size:
        print(f"⚠ WARNING: Found token ID {max_token_id} >= vocab_size {vocab_size}")
        print(f"  This will cause CUDA errors. Filtering dataset...")
        
        # Filter out problematic examples
        def is_valid(example):
            return all(tid < vocab_size for tid in example["input_ids"])
        
        tokenized_dataset = tokenized_dataset.filter(is_valid)
        print(f"✓ Dataset filtered to {len(tokenized_dataset)} valid examples")
    
    # Split into train and eval
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"✓ Train dataset size: {len(train_dataset)}")
    print(f"✓ Eval dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

# ============================================================
# Training
# ============================================================

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir, config, metrics_tracker):
    """Train the model with LoRA and track metrics"""
    
    # Create metrics callback
    metrics_callback = MetricsCallback()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy="steps",
        save_total_limit=config["save_total_limit"],
        fp16=config["fp16"],
        optim=config["optim"],
        remove_unused_columns=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        logging_first_step=True,
        ddp_find_unused_parameters=False,  # Improve performance
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[metrics_callback]
    )
    
    print(f"✓ Starting training...")
    metrics_tracker.start_time = time.time()
    
    trainer.train()
    
    metrics_tracker.end_time = time.time()
    metrics_tracker.callback_metrics = metrics_callback
    
    # Analyze LoRA updates
    metrics_tracker.final_metrics["lora_updates"] = metrics_tracker.analyze_lora_updates(model)
    
    print(f"✓ Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer

# ============================================================
# Testing
# ============================================================

# def test_model(model, tokenizer, prompt):
#     """Test the fine-tuned model"""
    
#     model.eval()
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             temperature=0.7,
#             do_sample=True,
#             top_p=0.9,
#             pad_token_id=tokenizer.pad_token_id
#         )
    
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

def test_model(model, tokenizer, prompt):
    """Test the fine-tuned model"""
    
    # CRITICAL: Re-enable cache for generation
    # It was disabled during training (line 692: model.config.use_cache = False)
    original_use_cache = model.config.use_cache
    model.config.use_cache = True
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Restore original setting (good practice)
    model.config.use_cache = original_use_cache
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    
# ============================================================
# Cleanup
# ============================================================

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================
# Comparative Analysis
# ============================================================

def generate_comparative_report(all_reports, output_dir):
    """Generate comparative analysis across all models"""
    
    if not all_reports:
        return
    
    # Helper function to convert numpy/torch types
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    comparison = {
        "models": list(all_reports.keys()),
        "summary": {}
    }
    
    # Compare key metrics
    for metric_name in ["training_time_seconds", "tokens_per_second", "peak_vram", 
                        "final_train_loss", "final_eval_loss", "convergence"]:
        comparison["summary"][metric_name] = {}
        
        for model_name, report in all_reports.items():
            if metric_name == "training_time_seconds":
                comparison["summary"][metric_name][model_name] = float(report.get("training_time_seconds", 0))
            elif metric_name == "tokens_per_second":
                comparison["summary"][metric_name][model_name] = float(report["throughput"].get("tokens_per_second", 0))
            elif metric_name == "peak_vram":
                comparison["summary"][metric_name][model_name] = float(report["gpu_vram"].get("peak_gb", 0))
            elif metric_name == "final_train_loss":
                losses = report["loss_curve"].get("train_losses", [])
                comparison["summary"][metric_name][model_name] = float(losses[-1]) if losses else None
            elif metric_name == "final_eval_loss":
                losses = report["loss_curve"].get("eval_losses", [])
                comparison["summary"][metric_name][model_name] = float(losses[-1]) if losses else None
            elif metric_name == "convergence":
                comparison["summary"][metric_name][model_name] = bool(report["convergence"].get("converged", False))
    
    # Convert entire comparison to native types
    comparison = convert_to_native(comparison)
    
    # Custom JSON encoder
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Save JSON
    with open(os.path.join(output_dir, "comparative_report.json"), "w") as f:
        json.dump(comparison, f, indent=2, cls=NumpyEncoder)
    
    # Generate text report
    with open(os.path.join(output_dir, "comparative_report.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("COMPARATIVE TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("TRAINING TIME\n")
        f.write("-"*80 + "\n")
        times = comparison["summary"]["training_time_seconds"]
        for model, time_sec in sorted(times.items(), key=lambda x: x[1]):
            f.write(f"{model:20s}: {time_sec/3600:.2f}h ({time_sec:.0f}s)\n")
        
        f.write("\nTHROUGHPUT (Tokens/Second)\n")
        f.write("-"*80 + "\n")
        throughput = comparison["summary"]["tokens_per_second"]
        for model, tps in sorted(throughput.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{model:20s}: {tps:.2f}\n")
        
        f.write("\nPEAK GPU VRAM (GB)\n")
        f.write("-"*80 + "\n")
        vram = comparison["summary"]["peak_vram"]
        for model, gb in sorted(vram.items(), key=lambda x: x[1]):
            f.write(f"{model:20s}: {gb:.2f}\n")
        
        f.write("\nFINAL TRAIN LOSS\n")
        f.write("-"*80 + "\n")
        train_loss = {k: v for k, v in comparison["summary"]["final_train_loss"].items() if v is not None}
        for model, loss in sorted(train_loss.items(), key=lambda x: x[1]):
            f.write(f"{model:20s}: {loss:.4f}\n")
        
        f.write("\nFINAL EVAL LOSS\n")
        f.write("-"*80 + "\n")
        eval_loss = {k: v for k, v in comparison["summary"]["final_eval_loss"].items() if v is not None}
        for model, loss in sorted(eval_loss.items(), key=lambda x: x[1]):
            f.write(f"{model:20s}: {loss:.4f}\n")
        
        f.write("\nCONVERGENCE STATUS\n")
        f.write("-"*80 + "\n")
        convergence = comparison["summary"]["convergence"]
        for model, converged in convergence.items():
            status = "✓ Converged" if converged else "✗ Not Converged"
            f.write(f"{model:20s}: {status}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    # Generate comparative plots
    _plot_comparative_metrics(all_reports, output_dir)

def _plot_comparative_metrics(all_reports, output_dir):
    """Generate comparative plots"""
    
    try:
        # Plot 1: Training time comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training time
        models = list(all_reports.keys())
        times = [all_reports[m]["training_time_seconds"]/3600 for m in models]
        axes[0, 0].barh(models, times, color='steelblue')
        axes[0, 0].set_xlabel('Hours')
        axes[0, 0].set_title('Training Time Comparison')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Throughput
        throughput = [all_reports[m]["throughput"]["tokens_per_second"] for m in models]
        axes[0, 1].barh(models, throughput, color='coral')
        axes[0, 1].set_xlabel('Tokens/Second')
        axes[0, 1].set_title('Throughput Comparison')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # VRAM
        vram = [all_reports[m]["gpu_vram"]["peak_gb"] for m in models]
        axes[1, 0].barh(models, vram, color='lightgreen')
        axes[1, 0].set_xlabel('GB')
        axes[1, 0].set_title('Peak GPU VRAM')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Final losses
        train_losses = [all_reports[m]["loss_curve"]["train_losses"][-1] 
                       if all_reports[m]["loss_curve"]["train_losses"] else 0 for m in models]
        eval_losses = [all_reports[m]["loss_curve"]["eval_losses"][-1] 
                      if all_reports[m]["loss_curve"]["eval_losses"] else 0 for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        axes[1, 1].bar(x - width/2, train_losses, width, label='Train', color='steelblue')
        axes[1, 1].bar(x + width/2, eval_losses, width, label='Eval', color='coral')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Final Loss Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparative_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate comparative plots: {e}")

# ============================================================
# Main Training Pipeline
# ============================================================

def train_single_model(model_path, model_info, train_dataset, eval_dataset, config):
    """Train a single model and track metrics"""
    
    try:
        # Create output directory
        output_dir = os.path.join(config["output_base_dir"], model_info["name"])
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics tracker
        metrics_tracker = TrainingMetrics(model_info["name"])
        
        # Store configuration
        metrics_tracker.lora_params = {
            "r": config["lora_r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "target_modules": model_info["target_modules"]
        }
        
        metrics_tracker.dataset_stats = {
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "batch_size": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
            "seq_length": config["max_seq_length"]
        }
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path, config)
        
        # Store model config
        metrics_tracker.model_config = {
            "model_path": model_path,
            "model_name": model_info["name"],
            "quantization": "4-bit" if config["use_4bit"] else "FP16",
            "vocab_size": model.config.vocab_size if hasattr(model.config, 'vocab_size') else None,
            "tokenizer_vocab_size": len(tokenizer),
            "hidden_size": model.config.hidden_size if hasattr(model.config, 'hidden_size') else None,
            "num_layers": model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else None,
            "num_attention_heads": model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else None,
        }
        
        # Get trainable parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0
        
        metrics_tracker.model_config["trainable_params"] = trainable_params
        metrics_tracker.model_config["all_params"] = all_params
        metrics_tracker.model_config["trainable_percent"] = trainable_percent
        
        # Apply LoRA
        model = setup_lora(model, model_info["target_modules"], config)
        
        # Train
        trainer = train_model(model, tokenizer, train_dataset, eval_dataset, 
                            output_dir, config, metrics_tracker)
        
        # Save metrics report
        print(f"\n{'='*70}")
        print(f"Generating metrics report for {model_info['name']}...")
        print(f"{'='*70}")
        
        report = metrics_tracker.save_report(output_dir)
        
        # Test
        print(f"\n{'='*70}")
        print(f"Testing {model_info['name']}...")
        print(f"{'='*70}")
        test_prompt = "### Instruction:\nWhat is machine learning?\n\n### Response:\n"
        response = test_model(model, tokenizer, test_prompt)
        print(f"Response: {response[:200]}...")
        
        # Cleanup
        del model
        del tokenizer
        del trainer
        cleanup_memory()
        
        print(f"\n✓ Successfully trained {model_info['name']}")
        print(f"✓ Model saved to: {output_dir}")
        print(f"✓ Metrics saved to: {output_dir}/training_metrics.json\n")
        
        return True, report
        
    except Exception as e:
        print(f"\n✗ Error training {model_info['name']}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False, None

def main():
    """Main training pipeline for all models"""
    
    print("="*70)
    print("Multi-Model LoRA Fine-tuning Script")
    print("="*70)
    
    # Determine which models to train
    train_mode = CONFIG.get("train_mode", "all")
    
    if isinstance(train_mode, list):
        # Train only specified models
        models_to_train = {k: v for k, v in MODELS.items() if v["name"] in train_mode}
        if not models_to_train:
            print(f"Error: No models found matching {train_mode}")
            print(f"Available models: {[v['name'] for v in MODELS.values()]}")
            return
        print(f"Training mode: SELECTIVE - {len(models_to_train)} models")
        for name in [v["name"] for v in models_to_train.values()]:
            print(f"  - {name}")
    elif train_mode == "all" or train_mode == "sequential":
        models_to_train = MODELS
        print(f"Training mode: ALL - {len(models_to_train)} models sequentially")
    else:
        print(f"Error: Invalid train_mode '{train_mode}'")
        print("Valid options: 'all', 'sequential', or list of model names")
        return
    
    # Check for completed models
    if CONFIG.get("skip_completed", False):
        remaining_models = {}
        for model_path, model_info in models_to_train.items():
            output_dir = os.path.join(CONFIG["output_base_dir"], model_info["name"])
            metrics_file = os.path.join(output_dir, "training_metrics.json")
            if os.path.exists(metrics_file):
                print(f"⊙ Skipping {model_info['name']} (already completed)")
            else:
                remaining_models[model_path] = model_info
        models_to_train = remaining_models
        print(f"Remaining models to train: {len(models_to_train)}")
    
    if not models_to_train:
        print("No models to train!")
        return
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Quantization: {'4-bit (QLoRA)' if CONFIG['use_4bit'] else 'FP16'}")
    print(f"  LoRA rank (r): {CONFIG['lora_r']}")
    print(f"  LoRA alpha: {CONFIG['lora_alpha']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Batch size: {CONFIG['per_device_train_batch_size']}")
    print(f"  Epochs: {CONFIG['num_train_epochs']}")
    print(f"  Output directory: {CONFIG['output_base_dir']}")
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Prepare dataset once (shared across all models)
    print(f"\n{'='*70}")
    print("Preparing shared dataset...")
    print(f"{'='*70}")
    
    # Use a temporary tokenizer to prepare dataset
    temp_model_path = list(models_to_train.keys())[0]
    print(f"✓ Loading tokenizer from: {temp_model_path}")
    
    temp_tokenizer = AutoTokenizer.from_pretrained(temp_model_path, trust_remote_code=True)
    if temp_tokenizer.pad_token is None:
        temp_tokenizer.pad_token = temp_tokenizer.eos_token
        temp_tokenizer.pad_token_id = temp_tokenizer.eos_token_id
    
    # Set model_max_length
    if temp_tokenizer.model_max_length is None or temp_tokenizer.model_max_length > 1e6:
        temp_tokenizer.model_max_length = CONFIG["max_seq_length"]
    
    train_dataset, eval_dataset = prepare_dataset(
        CONFIG,  # Pass entire config dict
        temp_tokenizer,
        CONFIG["max_seq_length"]
    )
    
    del temp_tokenizer
    cleanup_memory()
    
    # Train each model
    results = {}
    all_reports = {}
    
    print(f"\n{'='*70}")
    print(f"Starting training for {len(models_to_train)} model(s)...")
    print(f"{'='*70}\n")
    
    for idx, (model_path, model_info) in enumerate(models_to_train.items(), 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {idx}/{len(models_to_train)}: {model_info['name']}")
        print(f"{'#'*70}\n")
        
        success, report = train_single_model(
            model_path,
            model_info,
            train_dataset,
            eval_dataset,
            CONFIG
        )
        results[model_info["name"]] = success
        if report:
            all_reports[model_info["name"]] = report
        
        # Show progress
        completed = sum(results.values())
        print(f"\n>>> Progress: {completed}/{len(models_to_train)} models completed successfully\n")
    
    # Generate comparative report if multiple models were trained
    if len(all_reports) > 1:
        print(f"\n{'='*70}")
        print("Generating comparative analysis...")
        print(f"{'='*70}")
        generate_comparative_report(all_reports, CONFIG["output_base_dir"])
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {model_name}")
    
    successful = sum(results.values())
    print(f"\nCompleted: {successful}/{len(results)} models trained successfully")
    
    if len(all_reports) > 1:
        print(f"Comparative report saved to: {CONFIG['output_base_dir']}/comparative_report.txt")
    print("="*70)

def train_single_model_cli(model_name, dataset_file=None):
    """
    CLI function to train a single model
    
    Usage:
        python script.py --model gemma-7b --dataset train.jsonl
    """
    # Find model info
    model_info = None
    model_path = None
    for path, info in MODELS.items():
        if info["name"] == model_name:
            model_info = info
            model_path = path
            break
    
    if not model_info:
        print(f"Error: Model '{model_name}' not found")
        print(f"Available models: {[v['name'] for v in MODELS.values()]}")
        return
    
    # Update config
    if dataset_file:
        CONFIG["dataset_file"] = dataset_file
        CONFIG["dataset_name"] = None
    
    print(f"Training single model: {model_name}")
    
    # Prepare dataset
    temp_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if temp_tokenizer.pad_token is None:
        temp_tokenizer.pad_token = temp_tokenizer.eos_token
        temp_tokenizer.pad_token_id = temp_tokenizer.eos_token_id
    
    if temp_tokenizer.model_max_length is None or temp_tokenizer.model_max_length > 1e6:
        temp_tokenizer.model_max_length = CONFIG["max_seq_length"]
    
    train_dataset, eval_dataset = prepare_dataset(
        CONFIG,
        temp_tokenizer,
        CONFIG["max_seq_length"]
    )
    
    del temp_tokenizer
    cleanup_memory()
    
    # Train
    success, report = train_single_model(
        model_path,
        model_info,
        train_dataset,
        eval_dataset,
        CONFIG
    )
    
    if success:
        print(f"\n✓ Successfully trained {model_name}")
    else:
        print(f"\n✗ Failed to train {model_name}")

if __name__ == "__main__":
    main()