#!/usr/bin/env python3
"""
Convert DiffuCoder Dream model weights from PyTorch to JAX/Flax format.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.core import freeze
import pickle

# Add jax_lm to path
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

try:
    import torch
    from safetensors import safe_open
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch and safetensors available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch or safetensors not available")
    sys.exit(1)

from jax_lm.models.dream import DreamConfig, DreamForCausalLM, init_dream_model


def load_pytorch_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """Load PyTorch weights from safetensors files."""
    weights = {}
    
    # Check for safetensors index file
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        print("📋 Loading from safetensors index...")
        with open(index_file) as f:
            index = json.load(f)
        
        # Load weights from multiple safetensor files
        weight_map = index["weight_map"]
        files_to_load = set(weight_map.values())
        
        for file_name in files_to_load:
            file_path = model_path / file_name
            print(f"  Loading {file_name}...")
            
            with safe_open(str(file_path), framework="np") as f:
                for key in f.keys():
                    if key in weight_map and weight_map[key] == file_name:
                        weights[key] = f.get_tensor(key)
    else:
        # Try single safetensors file
        safetensors_file = model_path / "model.safetensors"
        if safetensors_file.exists():
            print("📋 Loading from single safetensors file...")
            with safe_open(str(safetensors_file), framework="np") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            raise FileNotFoundError("No safetensors files found")
    
    print(f"✅ Loaded {len(weights)} tensors")
    return weights


def convert_pytorch_to_jax_weights(pytorch_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert PyTorch weight names and formats to JAX/Flax."""
    jax_weights = {}
    
    print("🔄 Converting weight names and formats...")
    
    for pytorch_name, tensor in pytorch_weights.items():
        # Convert tensor to JAX array
        jax_tensor = jnp.array(tensor)
        
        # Convert PyTorch naming convention to Flax
        if pytorch_name == "model.embed_tokens.weight":
            jax_name = "DreamModel_0/Embed_0/embedding"
        elif pytorch_name.startswith("model.layers."):
            # Extract layer number
            parts = pytorch_name.split(".")
            layer_idx = parts[2]
            remaining = ".".join(parts[3:])
            
            # Map layer components
            if remaining == "input_layernorm.weight":
                jax_name = f"DreamModel_0/layers_{layer_idx}/DreamRMSNorm_0/weight"
            elif remaining == "post_attention_layernorm.weight":
                jax_name = f"DreamModel_0/layers_{layer_idx}/DreamRMSNorm_1/weight"
            elif remaining.startswith("self_attn."):
                attn_part = remaining[10:]  # Remove "self_attn."
                if attn_part == "q_proj.weight":
                    jax_name = f"DreamModel_0/layers_{layer_idx}/DreamAttention_0/q_proj/kernel"
                elif attn_part == "k_proj.weight":
                    jax_name = f"DreamModel_0/layers_{layer_idx}/DreamAttention_0/k_proj/kernel"
                elif attn_part == "v_proj.weight":
                    jax_name = f"DreamModel_0/layers_{layer_idx}/DreamAttention_0/v_proj/kernel"
                elif attn_part == "o_proj.weight":
                    jax_name = f"DreamModel_0/layers_{layer_idx}/DreamAttention_0/o_proj/kernel"
                else:
                    print(f"⚠️  Unknown attention weight: {pytorch_name}")
                    continue
            elif remaining.startswith("mlp."):
                mlp_part = remaining[4:]  # Remove "mlp."
                if mlp_part == "gate_proj.weight":
                    jax_name = f"DreamModel_0/layers_{layer_idx}/DreamMLP_0/Dense_0/kernel"
                elif mlp_part == "up_proj.weight":
                    jax_name = f"DreamModel_0/layers_{layer_idx}/DreamMLP_0/Dense_1/kernel"
                elif mlp_part == "down_proj.weight":
                    jax_name = f"DreamModel_0/layers_{layer_idx}/DreamMLP_0/Dense_2/kernel"
                else:
                    print(f"⚠️  Unknown MLP weight: {pytorch_name}")
                    continue
            else:
                print(f"⚠️  Unknown layer component: {pytorch_name}")
                continue
        elif pytorch_name == "model.norm.weight":
            jax_name = "DreamModel_0/DreamRMSNorm_0/weight"
        elif pytorch_name == "lm_head.weight":
            jax_name = "Dense_0/kernel"
        else:
            print(f"⚠️  Unknown weight: {pytorch_name}")
            continue
        
        # Transpose linear layer weights (PyTorch uses [out_features, in_features], Flax uses [in_features, out_features])
        if "kernel" in jax_name and jax_tensor.ndim == 2:
            jax_tensor = jax_tensor.T
        
        jax_weights[jax_name] = jax_tensor
        print(f"  {pytorch_name} -> {jax_name} {jax_tensor.shape}")
    
    return jax_weights


def create_flax_params_dict(converted_weights: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
    """Create properly nested Flax parameters dictionary."""
    params = {"params": {}}
    
    for full_name, tensor in converted_weights.items():
        # Split the path and create nested structure
        parts = full_name.split("/")
        current_dict = params["params"]
        
        # Navigate/create the nested structure
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        
        # Set the final tensor
        current_dict[parts[-1]] = tensor
    
    return freeze(params)


def test_model_loading(config: DreamConfig, params: Dict[str, Any]):
    """Test that the model can be initialized and run with converted weights."""
    print("🧪 Testing model loading...")
    
    try:
        # Initialize model
        rng = random.PRNGKey(42)
        model = DreamForCausalLM(config=config, dtype=config.dtype)
        
        # Test forward pass
        batch_size, seq_len = 1, 10
        dummy_input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        output = model.apply(params, dummy_input_ids, deterministic=True)
        
        print(f"✅ Model test successful!")
        print(f"   Input shape: {dummy_input_ids.shape}")
        print(f"   Output logits shape: {output['logits'].shape}")
        print(f"   Expected vocab size: {config.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Dream model weights to JAX/Flax")
    parser.add_argument(
        "--pytorch-model-path",
        default="./models/diffucoder-7b-complete",
        help="Path to PyTorch Dream model"
    )
    parser.add_argument(
        "--output-path", 
        default="./models/dream-jax",
        help="Output path for JAX model"
    )
    
    args = parser.parse_args()
    
    pytorch_model_path = Path(args.pytorch_model_path)
    output_path = Path(args.output_path)
    
    print("🔄 Dream Model PyTorch to JAX Conversion")
    print("=" * 50)
    
    if not pytorch_model_path.exists():
        print(f"❌ PyTorch model not found at {pytorch_model_path}")
        return
    
    try:
        # Load configuration
        print("📋 Loading model configuration...")
        config_file = pytorch_model_path / "config.json"
        with open(config_file) as f:
            config_dict = json.load(f)
        
        # Create JAX config
        dream_config = DreamConfig(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            rope_theta=config_dict["rope_theta"],
            rms_norm_eps=config_dict["rms_norm_eps"],
            attention_dropout=config_dict["attention_dropout"],
            mask_token_id=config_dict["mask_token_id"],
            pad_token_id=config_dict["pad_token_id"],
            bos_token_id=config_dict["bos_token_id"],
            eos_token_id=config_dict["eos_token_id"],
        )
        
        print(f"✅ Config loaded: {dream_config.num_hidden_layers} layers, {dream_config.hidden_size}d")
        
        # Load PyTorch weights
        pytorch_weights = load_pytorch_weights(pytorch_model_path)
        
        # Convert to JAX format
        converted_weights = convert_pytorch_to_jax_weights(pytorch_weights)
        
        # Create Flax parameters structure
        flax_params = create_flax_params_dict(converted_weights)
        
        # Test model loading
        if not test_model_loading(dream_config, flax_params):
            print("❌ Model test failed, aborting conversion")
            return
        
        # Save converted model
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save parameters
        params_file = output_path / "params.pkl"
        with open(params_file, "wb") as f:
            pickle.dump(flax_params, f)
        
        # Save config
        config_file = output_path / "config.json"
        config_dict_jax = {
            "vocab_size": dream_config.vocab_size,
            "hidden_size": dream_config.hidden_size,
            "intermediate_size": dream_config.intermediate_size,
            "num_hidden_layers": dream_config.num_hidden_layers,
            "num_attention_heads": dream_config.num_attention_heads,
            "num_key_value_heads": dream_config.num_key_value_heads,
            "max_position_embeddings": dream_config.max_position_embeddings,
            "rope_theta": dream_config.rope_theta,
            "rms_norm_eps": dream_config.rms_norm_eps,
            "attention_dropout": dream_config.attention_dropout,
            "mask_token_id": dream_config.mask_token_id,
            "pad_token_id": dream_config.pad_token_id,
            "bos_token_id": dream_config.bos_token_id,
            "eos_token_id": dream_config.eos_token_id,
        }
        
        with open(config_file, "w") as f:
            json.dump(config_dict_jax, f, indent=2)
        
        print(f"\n✅ Conversion completed!")
        print(f"📁 JAX model saved to: {output_path}")
        print(f"   Parameters: {params_file}")
        print(f"   Configuration: {config_file}")
        
        # Print model info
        total_params = sum(p.size for p in jax.tree_leaves(flax_params))
        print(f"\n📊 Model Info:")
        print(f"   Total parameters: {total_params:,} ({total_params/1e9:.1f}B)")
        print(f"   Hidden size: {dream_config.hidden_size}")
        print(f"   Layers: {dream_config.num_hidden_layers}")
        print(f"   Attention heads: {dream_config.num_attention_heads}")
        print(f"   Vocab size: {dream_config.vocab_size}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()