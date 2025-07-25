#!/usr/bin/env python3
"""Convert PyTorch DiffuCoder weights to JAX format."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint as ocp

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig


def load_pytorch_weights(pytorch_path: Path) -> Dict[str, torch.Tensor]:
    """Load PyTorch weights from checkpoint files."""
    weights = {}
    
    # Check for different file formats
    bin_files = list(pytorch_path.glob("*.bin"))
    if bin_files:
        print(f"Found {len(bin_files)} .bin files")
        for bin_file in bin_files:
            print(f"Loading {bin_file.name}...")
            state_dict = torch.load(bin_file, map_location="cpu")
            weights.update(state_dict)
    else:
        # Try loading from a single checkpoint
        ckpt_file = pytorch_path / "pytorch_model.bin"
        if ckpt_file.exists():
            print(f"Loading {ckpt_file}...")
            weights = torch.load(ckpt_file, map_location="cpu")
        else:
            raise ValueError(f"No PyTorch weights found in {pytorch_path}")
    
    return weights


def convert_attention_weights(
    pytorch_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    config: DiffuCoderConfig,
) -> Dict[str, jnp.ndarray]:
    """Convert attention layer weights from PyTorch to JAX format."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    jax_weights = {}
    
    # Query, Key, Value projections
    if f"{prefix}.q_proj.weight" in pytorch_weights:
        jax_weights["q_proj"] = {
            "kernel": jnp.array(pytorch_weights[f"{prefix}.q_proj.weight"].T.numpy())
        }
        if f"{prefix}.q_proj.bias" in pytorch_weights:
            jax_weights["q_proj"]["bias"] = jnp.array(
                pytorch_weights[f"{prefix}.q_proj.bias"].numpy()
            )
    
    if f"{prefix}.k_proj.weight" in pytorch_weights:
        jax_weights["k_proj"] = {
            "kernel": jnp.array(pytorch_weights[f"{prefix}.k_proj.weight"].T.numpy())
        }
        if f"{prefix}.k_proj.bias" in pytorch_weights:
            jax_weights["k_proj"]["bias"] = jnp.array(
                pytorch_weights[f"{prefix}.k_proj.bias"].numpy()
            )
    
    if f"{prefix}.v_proj.weight" in pytorch_weights:
        jax_weights["v_proj"] = {
            "kernel": jnp.array(pytorch_weights[f"{prefix}.v_proj.weight"].T.numpy())
        }
        if f"{prefix}.v_proj.bias" in pytorch_weights:
            jax_weights["v_proj"]["bias"] = jnp.array(
                pytorch_weights[f"{prefix}.v_proj.bias"].numpy()
            )
    
    # Output projection
    if f"{prefix}.o_proj.weight" in pytorch_weights:
        jax_weights["o_proj"] = {
            "kernel": jnp.array(pytorch_weights[f"{prefix}.o_proj.weight"].T.numpy())
        }
        if f"{prefix}.o_proj.bias" in pytorch_weights:
            jax_weights["o_proj"]["bias"] = jnp.array(
                pytorch_weights[f"{prefix}.o_proj.bias"].numpy()
            )
    
    return jax_weights


def convert_mlp_weights(
    pytorch_weights: Dict[str, torch.Tensor],
    layer_idx: int,
) -> Dict[str, jnp.ndarray]:
    """Convert MLP layer weights from PyTorch to JAX format."""
    prefix = f"model.layers.{layer_idx}.mlp"
    jax_weights = {}
    
    # Gate projection
    if f"{prefix}.gate_proj.weight" in pytorch_weights:
        jax_weights["gate_proj"] = {
            "kernel": jnp.array(pytorch_weights[f"{prefix}.gate_proj.weight"].T.numpy())
        }
    
    # Up projection
    if f"{prefix}.up_proj.weight" in pytorch_weights:
        jax_weights["up_proj"] = {
            "kernel": jnp.array(pytorch_weights[f"{prefix}.up_proj.weight"].T.numpy())
        }
    
    # Down projection
    if f"{prefix}.down_proj.weight" in pytorch_weights:
        jax_weights["down_proj"] = {
            "kernel": jnp.array(pytorch_weights[f"{prefix}.down_proj.weight"].T.numpy())
        }
    
    return jax_weights


def convert_weights(
    pytorch_weights: Dict[str, torch.Tensor],
    config: DiffuCoderConfig,
) -> Dict[str, Any]:
    """Convert all PyTorch weights to JAX format."""
    jax_params = {"params": {}}
    
    # Convert embeddings
    if "model.embed_tokens.weight" in pytorch_weights:
        embed_weight = pytorch_weights["model.embed_tokens.weight"].numpy()
        # Add mask and pad token embeddings if needed
        if embed_weight.shape[0] == config.vocab_size:
            # Add two extra embeddings for mask and pad tokens
            extra_embeds = np.random.normal(
                0, config.initializer_range, (2, embed_weight.shape[1])
            ).astype(embed_weight.dtype)
            embed_weight = np.concatenate([embed_weight, extra_embeds], axis=0)
        
        jax_params["params"]["DiffuCoderEmbedding_0"] = {
            "Embed_0": {"embedding": jnp.array(embed_weight)}
        }
    
    # Convert transformer layers
    for i in range(config.num_hidden_layers):
        layer_params = {}
        
        # RMSNorm before attention
        if f"model.layers.{i}.input_layernorm.weight" in pytorch_weights:
            layer_params["RMSNorm_0"] = {
                "weight": jnp.array(pytorch_weights[f"model.layers.{i}.input_layernorm.weight"].numpy())
            }
        
        # Self-attention
        attn_params = convert_attention_weights(pytorch_weights, i, config)
        if attn_params:
            layer_params["DiffuCoderAttention_0"] = attn_params
        
        # RMSNorm before MLP
        if f"model.layers.{i}.post_attention_layernorm.weight" in pytorch_weights:
            layer_params["RMSNorm_1"] = {
                "weight": jnp.array(pytorch_weights[f"model.layers.{i}.post_attention_layernorm.weight"].numpy())
            }
        
        # MLP
        mlp_params = convert_mlp_weights(pytorch_weights, i)
        if mlp_params:
            layer_params["DiffuCoderMLP_0"] = mlp_params
        
        jax_params["params"][f"layer_{i}"] = layer_params
    
    # Final RMSNorm
    if "model.norm.weight" in pytorch_weights:
        jax_params["params"]["norm"] = {
            "weight": jnp.array(pytorch_weights["model.norm.weight"].numpy())
        }
    
    # Language modeling head
    if "lm_head.weight" in pytorch_weights:
        lm_head_weight = pytorch_weights["lm_head.weight"].numpy()
        # Add extra dimensions if needed
        if lm_head_weight.shape[0] == config.vocab_size:
            extra_weights = np.random.normal(
                0, config.initializer_range, (2, lm_head_weight.shape[1])
            ).astype(lm_head_weight.dtype)
            lm_head_weight = np.concatenate([lm_head_weight, extra_weights], axis=0)
        
        jax_params["params"]["lm_head"] = {
            "kernel": jnp.array(lm_head_weight.T)  # Transpose for JAX Dense layer
        }
    
    return jax_params


def save_jax_checkpoint(params: Dict[str, Any], output_path: Path):
    """Save JAX parameters using Orbax."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save using Orbax
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(output_path / "checkpoint", params)
    
    print(f"Saved JAX checkpoint to {output_path / 'checkpoint'}")


def convert_pytorch_to_jax(
    pytorch_path: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
):
    """Main conversion function."""
    # Load config
    if config_path:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = DiffuCoderConfig(**config_dict)
    else:
        # Try to load from PyTorch directory
        pytorch_config_path = pytorch_path / "config.json"
        if pytorch_config_path.exists():
            with open(pytorch_config_path, "r") as f:
                pytorch_config = json.load(f)
            
            # Map PyTorch config to JAX config
            config = DiffuCoderConfig(
                vocab_size=pytorch_config.get("vocab_size", 32000),
                hidden_size=pytorch_config.get("hidden_size", 4096),
                intermediate_size=pytorch_config.get("intermediate_size", 11008),
                num_hidden_layers=pytorch_config.get("num_hidden_layers", 32),
                num_attention_heads=pytorch_config.get("num_attention_heads", 32),
                num_key_value_heads=pytorch_config.get("num_key_value_heads"),
                hidden_act=pytorch_config.get("hidden_act", "silu"),
                max_position_embeddings=pytorch_config.get("max_position_embeddings", 4096),
                initializer_range=pytorch_config.get("initializer_range", 0.02),
                rms_norm_eps=pytorch_config.get("rms_norm_eps", 1e-5),
                rope_theta=pytorch_config.get("rope_theta", 10000.0),
                attention_bias=pytorch_config.get("attention_bias", False),
                attention_dropout=pytorch_config.get("attention_dropout", 0.0),
            )
        else:
            raise ValueError("No config file found. Please provide --config-path")
    
    # Save config
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Load PyTorch weights
    print("Loading PyTorch weights...")
    pytorch_weights = load_pytorch_weights(pytorch_path)
    print(f"Loaded {len(pytorch_weights)} weight tensors")
    
    # Convert weights
    print("Converting weights to JAX format...")
    jax_params = convert_weights(pytorch_weights, config)
    
    # Save JAX checkpoint
    print("Saving JAX checkpoint...")
    save_jax_checkpoint(jax_params, output_path)
    
    # Save metadata
    metadata = {
        "source_model": str(pytorch_path),
        "config": config.__dict__,
        "conversion_complete": True,
    }
    with open(output_path / "conversion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Conversion complete! JAX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch DiffuCoder weights to JAX format"
    )
    parser.add_argument(
        "--pytorch-path",
        type=Path,
        required=True,
        help="Path to PyTorch model weights directory",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save JAX model weights",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to model config JSON (optional, will try to load from PyTorch dir)",
    )
    
    args = parser.parse_args()
    convert_pytorch_to_jax(args.pytorch_path, args.output_path, args.config_path)


if __name__ == "__main__":
    main()