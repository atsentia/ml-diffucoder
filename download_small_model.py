#!/usr/bin/env python3
"""Download a small model for testing JAX DiffuCoder."""

import os
import sys
import subprocess
from pathlib import Path

def check_huggingface_cli():
    """Check if huggingface-cli is available."""
    try:
        result = subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, text=True)
        print(f"HuggingFace CLI version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("huggingface-cli not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub[cli]"])
        return True

def download_model_info():
    """Try to download just model configuration."""
    print("\n=== Downloading Model Information ===")
    
    # Create directory
    models_dir = Path("models/test")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download just the config file
    print("Attempting to download model configuration...")
    
    script = """
import json
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

model_id = "apple/DiffuCoder-7B-Base"
output_dir = Path("models/test")
output_dir.mkdir(parents=True, exist_ok=True)

try:
    # Try to list files in the repository
    print(f"Checking files in {model_id}...")
    files = list(list_repo_files(model_id))
    
    config_files = [f for f in files if "config" in f.lower()]
    print(f"Found config files: {config_files}")
    
    # Try to download config.json
    if "config.json" in files:
        print("Downloading config.json...")
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            cache_dir=output_dir / "cache",
            local_dir=output_dir,
        )
        print(f"Config downloaded to: {config_path}")
        
        # Load and display config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print("\\nModel Configuration:")
        for key in ["model_type", "hidden_size", "num_hidden_layers", "vocab_size"]:
            if key in config:
                print(f"  {key}: {config[key]}")
        
        # Save minimal config for testing
        minimal_config = {
            "model_type": config.get("model_type", "llama"),
            "vocab_size": 1000,  # Small for testing
            "hidden_size": 256,  # Small for testing
            "num_hidden_layers": 2,  # Very small
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "max_position_embeddings": 512,
            "rms_norm_eps": config.get("rms_norm_eps", 1e-5),
        }
        
        with open(output_dir / "minimal_config.json", "w") as f:
            json.dump(minimal_config, f, indent=2)
        
        print("\\nCreated minimal config for testing")
        
except Exception as e:
    print(f"Error accessing model: {e}")
    print("\\nCreating default test configuration...")
    
    # Create a default config for testing
    default_config = {
        "model_type": "diffucoder",
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
    }
    
    with open(output_dir / "minimal_config.json", "w") as f:
        json.dump(default_config, f, indent=2)
    
    print("Created default test configuration")
"""
    
    result = subprocess.run([sys.executable, "-c", script], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
    
    return result.returncode == 0

def create_mock_weights():
    """Create small mock weights for testing."""
    print("\n=== Creating Mock Weights for Testing ===")
    
    script = """
import torch
import json
import numpy as np
from pathlib import Path

# Load config
config_path = Path("models/test/minimal_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

print(f"Creating mock weights with config:")
print(f"  Layers: {config['num_hidden_layers']}")
print(f"  Hidden size: {config['hidden_size']}")
print(f"  Vocab size: {config['vocab_size']}")

# Create minimal weights
state_dict = {}

# Embeddings
state_dict["model.embed_tokens.weight"] = torch.randn(
    config["vocab_size"] + 2, config["hidden_size"]
) * 0.02

# Transformer layers (minimal)
for i in range(config["num_hidden_layers"]):
    prefix = f"model.layers.{i}"
    
    # Attention
    hidden = config["hidden_size"]
    state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden) * 0.02
    state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden, hidden) * 0.02
    state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden, hidden) * 0.02
    state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden) * 0.02
    
    # MLP
    intermediate = config["intermediate_size"]
    state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate, hidden) * 0.02
    state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(intermediate, hidden) * 0.02
    state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden, intermediate) * 0.02
    
    # Norms
    state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden)
    state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden)

# Final norm and head
state_dict["model.norm.weight"] = torch.ones(config["hidden_size"])
state_dict["lm_head.weight"] = torch.randn(config["vocab_size"], config["hidden_size"]) * 0.02

# Save
output_path = Path("models/test/pytorch_model.bin")
torch.save(state_dict, output_path)

# Calculate size
total_params = sum(p.numel() for p in state_dict.values())
size_mb = sum(p.numel() * 4 for p in state_dict.values()) / 1024 / 1024

print(f"\\nCreated mock model:")
print(f"  Total parameters: {total_params / 1e6:.2f}M")
print(f"  File size: {size_mb:.1f} MB")
print(f"  Saved to: {output_path}")

# Also save config in the right format
torch_config = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": config["vocab_size"],
    "hidden_size": config["hidden_size"],
    "intermediate_size": config["intermediate_size"],
    "num_hidden_layers": config["num_hidden_layers"],
    "num_attention_heads": config["num_attention_heads"],
    "max_position_embeddings": config["max_position_embeddings"],
    "rms_norm_eps": config["rms_norm_eps"],
    "torch_dtype": "float32",
}

with open(Path("models/test/config.json"), "w") as f:
    json.dump(torch_config, f, indent=2)
"""
    
    # Check if torch is available
    try:
        import torch
        subprocess.run([sys.executable, "-c", script], check=True)
        return True
    except ImportError:
        print("PyTorch not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch"], check=True)
        subprocess.run([sys.executable, "-c", script], check=True)
        return True
    except Exception as e:
        print(f"Error creating mock weights: {e}")
        return False

def main():
    print("=" * 60)
    print("DiffuCoder Model Download for Testing")
    print("=" * 60)
    
    # Check HuggingFace CLI
    if not check_huggingface_cli():
        print("Failed to setup HuggingFace CLI")
        return
    
    # Try to download model info
    if download_model_info():
        print("\n✓ Successfully downloaded model information")
    else:
        print("\n✗ Could not download model information")
    
    # Create mock weights for testing
    if create_mock_weights():
        print("\n✓ Successfully created test weights")
        
        print("\n" + "=" * 60)
        print("Ready for Testing!")
        print("=" * 60)
        print("\nYou can now run:")
        print("1. Convert to JAX format:")
        print("   python jax_lm/scripts/convert_pytorch_to_jax.py \\")
        print("     --pytorch-path ./models/test \\")
        print("     --output-path ./models/jax_test")
        print("\n2. Run benchmark:")
        print("   python jax_lm/benchmarks/hardware_benchmark.py \\")
        print("     --backend cpu --model-size small")
        
        print("\nNote: This is a tiny test model (256 hidden size, 2 layers)")
        print("Full model would be ~26GB (4096 hidden size, 32 layers)")
    else:
        print("\n✗ Failed to create test weights")

if __name__ == "__main__":
    main()