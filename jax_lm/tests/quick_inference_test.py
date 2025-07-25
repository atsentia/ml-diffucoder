#!/usr/bin/env python3
"""Quick test to verify JAX inference works with actual model weights.

This is a minimal test that loads the model and runs a single inference.
"""

import sys
import time
from pathlib import Path

def test_inference():
    """Test basic inference with real weights."""
    print("🧪 JAX DiffuCoder Quick Inference Test")
    print("=" * 50)
    
    model_path = Path("./models/dream-jax")
    tokenizer_path = Path("./models/diffucoder-7b-complete")
    
    # Check if model exists
    if not model_path.exists():
        print("❌ Model not found at:", model_path)
        print("   Please ensure JAX model weights are downloaded")
        return False
        
    if not (model_path / "params.pkl").exists():
        print("❌ params.pkl not found in:", model_path)
        return False
        
    print("✅ Model files found")
    
    # Try to load model
    try:
        print("\n📦 Loading model...")
        start = time.time()
        
        from jax_lm.inference import DiffuCoderInference
        import jax.numpy as jnp
        
        model = DiffuCoderInference(
            model_path=str(model_path),
            tokenizer_path=str(tokenizer_path) if tokenizer_path.exists() else None,
            dtype=jnp.float32,  # Use float32 for CPU compatibility
        )
        
        load_time = time.time() - start
        print(f"✅ Model loaded in {load_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Try inference
    try:
        print("\n🎯 Running inference...")
        prompt = "def hello_world():"
        
        start = time.time()
        output = model.generate(
            prompt,
            max_new_tokens=30,
            temperature=0.3,
            num_steps=10,  # Very few steps for quick test
        )
        gen_time = time.time() - start
        
        print(f"✅ Generation completed in {gen_time:.1f}s")
        print(f"\n📝 Input: {prompt}")
        print(f"📄 Output: {output}")
        
        # Basic validation
        assert len(output) > len(prompt), "No tokens generated"
        assert "def" in output, "Output should contain function definition"
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick JAX inference test")
    parser.add_argument("--model-path", default="./models/dream-jax", 
                       help="Path to JAX model")
    parser.add_argument("--tokenizer-path", default="./models/diffucoder-7b-complete",
                       help="Path to tokenizer")
    args = parser.parse_args()
    
    # Override paths if provided
    if args.model_path != "./models/dream-jax":
        import os
        os.environ["MODEL_PATH"] = args.model_path
    
    success = test_inference()
    sys.exit(0 if success else 1)