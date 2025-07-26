#!/usr/bin/env python3
"""Test loading model from HuggingFace after upload."""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

print("Testing jax-diffucoder with HuggingFace model...")

# Test 1: Package import
try:
    import jax_lm
    print(f"✅ Package imported: version {jax_lm.__version__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Load model from HuggingFace
try:
    from jax_lm import load_model, generate
    
    print("\nLoading model from HuggingFace...")
    model, params, tokenizer = load_model("atsentia/DiffuCoder-7B-JAX")
    print("✅ Model loaded successfully!")
    
    # Test 3: Simple generation
    prompt = "def hello_world():"
    print(f"\nGenerating code for prompt: '{prompt}'")
    
    output = generate(
        model, params, prompt, tokenizer,
        max_new_tokens=50,
        temperature=0.7,
        seed=42
    )
    
    print(f"\nGenerated output:\n{output}")
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Model loading/generation failed: {e}")
    import traceback
    traceback.print_exc()