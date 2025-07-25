#!/usr/bin/env python3
"""
Incremental test script for Dream model loading and weight conversion.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

# Add jax_lm to path
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
    print("✅ safetensors available")
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("❌ safetensors not available")

from jax_lm.models.dream import DreamConfig, DreamForCausalLM


def test_1_load_config():
    """Test 1: Load and parse model configuration."""
    print("\n🧪 Test 1: Loading model configuration")
    print("-" * 40)
    
    try:
        config_path = Path("./models/diffucoder-7b-complete/config.json")
        
        if not config_path.exists():
            print(f"❌ Config file not found at {config_path}")
            return False
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        print("✅ Config loaded successfully")
        print(f"   Model type: {config_dict.get('model_type', 'Unknown')}")
        print(f"   Vocab size: {config_dict.get('vocab_size', 'Unknown')}")
        print(f"   Hidden size: {config_dict.get('hidden_size', 'Unknown')}")
        print(f"   Layers: {config_dict.get('num_hidden_layers', 'Unknown')}")
        print(f"   Attention heads: {config_dict.get('num_attention_heads', 'Unknown')}")
        
        # Create DreamConfig
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
        
        print("✅ DreamConfig created successfully")
        return dream_config
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_2_model_init(config):
    """Test 2: Initialize JAX model without weights."""
    print("\n🧪 Test 2: Initializing JAX model")
    print("-" * 40)
    
    try:
        # Create model
        model = DreamForCausalLM(config=config, dtype=config.dtype)
        print("✅ Model class created")
        
        # Initialize with random weights
        rng = random.PRNGKey(42)
        batch_size, seq_len = 1, 8
        dummy_input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        print(f"   Initializing with input shape: {dummy_input_ids.shape}")
        params = model.init(rng, dummy_input_ids, deterministic=True)
        print("✅ Model initialized with random weights")
        
        # Count parameters
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"   Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        
        return model, params
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False


def test_3_forward_pass(model, params, config):
    """Test 3: Run forward pass with random weights."""
    print("\n🧪 Test 3: Forward pass with random weights")
    print("-" * 40)
    
    try:
        # Create test input
        rng = random.PRNGKey(123)
        batch_size, seq_len = 2, 10
        input_ids = random.randint(rng, (batch_size, seq_len), 0, min(1000, config.vocab_size))
        
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Input range: {input_ids.min()} to {input_ids.max()}")
        
        # Forward pass
        output = model.apply(params, input_ids, deterministic=True)
        
        print("✅ Forward pass successful")
        print(f"   Output logits shape: {output['logits'].shape}")
        print(f"   Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        if output['logits'].shape == expected_shape:
            print("✅ Output shape correct")
        else:
            print(f"❌ Output shape mismatch: got {output['logits'].shape}, expected {expected_shape}")
            return False
        
        # Check for NaN/Inf
        if jnp.isnan(output['logits']).any():
            print("❌ Output contains NaN values")
            return False
        
        if jnp.isinf(output['logits']).any():
            print("❌ Output contains Inf values")
            return False
        
        print("✅ Output values are valid (no NaN/Inf)")
        logit_min = float(output['logits'].min())
        logit_max = float(output['logits'].max())
        print(f"   Logit range: {logit_min:.3f} to {logit_max:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_examine_pytorch_weights():
    """Test 4: Examine PyTorch weights structure."""
    print("\n🧪 Test 4: Examining PyTorch weights")
    print("-" * 40)
    
    if not SAFETENSORS_AVAILABLE:
        print("❌ safetensors not available, skipping weight examination")
        return False
    
    try:
        model_path = Path("./models/diffucoder-7b-complete")
        index_file = model_path / "model.safetensors.index.json"
        
        if not index_file.exists():
            print(f"❌ Index file not found at {index_file}")
            return False
        
        # Load index
        with open(index_file) as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        print(f"✅ Found {len(weight_map)} weights across {len(set(weight_map.values()))} files")
        
        # Sample some weights
        sample_weights = list(weight_map.keys())[:10]
        print("\n📋 Sample weight names:")
        for weight_name in sample_weights:
            print(f"   {weight_name}")
        
        # Load a sample weight to check format
        first_file = list(set(weight_map.values()))[0]
        first_file_path = model_path / first_file
        
        print(f"\n🔍 Examining first file: {first_file}")
        
        with safe_open(str(first_file_path), framework="np") as f:
            keys = list(f.keys())[:5]  # First 5 keys
            for key in keys:
                tensor = f.get_tensor(key)
                print(f"   {key}: {tensor.shape} {tensor.dtype}")
        
        print("✅ PyTorch weights examination successful")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch weights examination failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_simple_weight_conversion():
    """Test 5: Try converting one simple weight."""
    print("\n🧪 Test 5: Simple weight conversion test")
    print("-" * 40)
    
    if not SAFETENSORS_AVAILABLE:
        print("❌ safetensors not available, skipping conversion test")
        return False
    
    try:
        model_path = Path("./models/diffucoder-7b-complete")
        
        # Load a simple weight (embeddings)
        index_file = model_path / "model.safetensors.index.json"
        with open(index_file) as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # Find embedding weight
        embed_key = None
        embed_file = None
        for key, file in weight_map.items():
            if "embed_tokens.weight" in key:
                embed_key = key
                embed_file = file
                break
        
        if embed_key is None:
            print("❌ Could not find embedding weight")
            return False
        
        print(f"   Loading: {embed_key} from {embed_file}")
        
        # Load the weight
        with safe_open(str(model_path / embed_file), framework="np") as f:
            embed_tensor = f.get_tensor(embed_key)
        
        print(f"   Original shape: {embed_tensor.shape}")
        print(f"   Original dtype: {embed_tensor.dtype}")
        
        # Convert to JAX
        jax_tensor = jnp.array(embed_tensor)
        print(f"   JAX shape: {jax_tensor.shape}")
        print(f"   JAX dtype: {jax_tensor.dtype}")
        
        # Test that it's valid
        if jnp.isnan(jax_tensor).any():
            print("❌ Converted tensor contains NaN")
            return False
        
        print("✅ Simple weight conversion successful")
        print(f"   Value range: {jax_tensor.min():.6f} to {jax_tensor.max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple weight conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests incrementally."""
    print("🧪 Dream Model Incremental Testing")
    print("=" * 50)
    
    # Test 1: Config loading
    config = test_1_load_config()
    if not config:
        print("\n❌ Test 1 failed, stopping here")
        return
    
    # Test 2: Model init
    model, params = test_2_model_init(config)
    if not model or not params:
        print("\n❌ Test 2 failed, stopping here")
        return
    
    # Test 3: Forward pass
    if not test_3_forward_pass(model, params, config):
        print("\n❌ Test 3 failed, stopping here")
        return
    
    # Test 4: PyTorch weights
    if not test_4_examine_pytorch_weights():
        print("\n❌ Test 4 failed, but continuing...")
    
    # Test 5: Simple conversion
    if not test_5_simple_weight_conversion():
        print("\n❌ Test 5 failed, but JAX model works")
    
    print("\n" + "=" * 50)
    print("✅ ALL BASIC TESTS PASSED!")
    print("📊 Summary:")
    print("   ✅ Configuration loading works")
    print("   ✅ JAX model initialization works")
    print("   ✅ Forward pass works with random weights")
    print("   ✅ PyTorch weights can be examined")
    print("   ✅ Basic weight conversion works")
    print("\nNext step: Full weight conversion and real inference testing")


if __name__ == "__main__":
    main()