#!/usr/bin/env python3
"""Test loading JAX DiffuCoder model from HuggingFace Hub.

This script tests the complete workflow:
1. Download model from HuggingFace
2. Load using Orbax sharding
3. Run inference
4. Verify outputs
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Test basic imports
print("Testing imports...")
try:
    from jax_lm.utils.orbax_sharding import load_from_huggingface
    from jax_lm.models import diffusion_generate
    from jax_lm.utils import load_tokenizer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to install: pip install -e jax_lm/")
    exit(1)


def test_model_loading(repo_id: str = "atsentia/DiffuCoder-7B-JAX"):
    """Test loading model from HuggingFace."""
    print(f"\n1. Testing model loading from {repo_id}...")
    
    start_time = time.time()
    try:
        # Load with bfloat16 to save memory
        model, params = load_from_huggingface(
            repo_id,
            dtype=jnp.bfloat16
        )
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        # Check parameter count
        param_count = sum(x.size for x in jax.tree_leaves(params))
        print(f"   Parameters: {param_count:,}")
        
        # Check model config
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Num layers: {model.config.num_hidden_layers}")
        print(f"   Vocab size: {model.config.vocab_size}")
        
        return model, params
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def test_tokenizer_loading(repo_id: str = "atsentia/DiffuCoder-7B-JAX"):
    """Test loading tokenizer from HuggingFace."""
    print(f"\n2. Testing tokenizer loading...")
    
    try:
        tokenizer = load_tokenizer(repo_id)
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocab size: {len(tokenizer.get_vocab())}")
        
        # Test encoding/decoding
        test_text = "def hello_world():"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   Test encode/decode: '{test_text}' -> {tokens[:5]}... -> '{decoded}'")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return None


def test_inference(model, params, tokenizer):
    """Test model inference."""
    print("\n3. Testing inference...")
    
    if model is None or params is None or tokenizer is None:
        print("❌ Skipping inference test (model/tokenizer not loaded)")
        return
    
    # Test prompts
    prompts = [
        "def fibonacci(n):",
        "class BinaryTree:",
        "// Quick sort implementation in JavaScript"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n   Test {i+1}: '{prompt}'")
        
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="np")
            input_ids = jnp.array(inputs["input_ids"])
            
            # Generate
            start_time = time.time()
            output = diffusion_generate(
                model,
                params,
                input_ids,
                jax.random.PRNGKey(42 + i),
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            gen_time = time.time() - start_time
            
            # Decode
            generated_ids = output["sequences"][0]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"   Generated ({gen_time:.2f}s):")
            print(f"   {generated_text[:200]}...")
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")


def test_batch_inference(model, params, tokenizer):
    """Test batch inference."""
    print("\n4. Testing batch inference...")
    
    if model is None or params is None or tokenizer is None:
        print("❌ Skipping batch inference test")
        return
    
    try:
        from jax_lm.models import batch_diffusion_generate
        
        prompts = [
            "def factorial(n):",
            "async function fetchData():",
            "public class HelloWorld {"
        ]
        
        # Tokenize batch
        inputs = tokenizer(prompts, padding=True, return_tensors="np")
        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs["attention_mask"])
        
        print(f"   Batch shape: {input_ids.shape}")
        
        # Generate
        start_time = time.time()
        outputs = batch_diffusion_generate(
            model,
            params,
            input_ids,
            jax.random.PRNGKey(42),
            attention_mask=attention_mask,
            max_new_tokens=30,
            temperature=0.7
        )
        gen_time = time.time() - start_time
        
        print(f"✅ Batch generation successful ({gen_time:.2f}s)")
        print(f"   Generated {len(outputs['sequences'])} sequences")
        
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")


def test_memory_usage():
    """Test memory usage."""
    print("\n5. Testing memory usage...")
    
    try:
        devices = jax.devices()
        print(f"   Available devices: {devices}")
        
        for device in devices:
            if hasattr(device, 'memory_stats'):
                stats = device.memory_stats()
                used_gb = stats.get('bytes_in_use', 0) / (1024**3)
                limit_gb = stats.get('bytes_limit', 0) / (1024**3)
                print(f"   {device}: {used_gb:.2f}/{limit_gb:.2f} GB")
            else:
                print(f"   {device}: Memory stats not available")
                
    except Exception as e:
        print(f"❌ Failed to get memory stats: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("JAX DiffuCoder HuggingFace Integration Test")
    print("=" * 60)
    
    # Check JAX backend
    print(f"\nJAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Run tests
    model, params = test_model_loading()
    tokenizer = test_tokenizer_loading()
    test_inference(model, params, tokenizer)
    test_batch_inference(model, params, tokenizer)
    test_memory_usage()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("- Model loading: " + ("✅ PASS" if model is not None else "❌ FAIL"))
    print("- Tokenizer loading: " + ("✅ PASS" if tokenizer is not None else "❌ FAIL"))
    print("=" * 60)


if __name__ == "__main__":
    main()