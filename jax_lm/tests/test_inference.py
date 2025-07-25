#!/usr/bin/env python3
"""Minimal test suite for JAX DiffuCoder inference."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np


def test_jax_setup():
    """Test JAX is properly configured."""
    print("Testing JAX setup...")
    
    # Check JAX version
    print(f"  JAX version: {jax.__version__}")
    
    # Check devices
    devices = jax.devices()
    print(f"  Available devices: {devices}")
    print(f"  Default backend: {jax.default_backend()}")
    
    # Simple computation test
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    z = x + y
    
    assert jnp.array_equal(z, jnp.array([5, 7, 9])), "Basic JAX computation failed"
    print("  ✅ JAX setup OK")
    return True


def test_model_imports():
    """Test model imports work correctly."""
    print("\nTesting model imports...")
    
    try:
        from jax_lm import DiffuCoder, DiffuCoderConfig
        print("  ✅ Main imports OK")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    
    try:
        from jax_lm.models.dream import DreamForCausalLM, DreamConfig
        print("  ✅ Model imports OK")
    except ImportError as e:
        print(f"  ❌ Model import error: {e}")
        return False
    
    try:
        from jax_lm.generate_diffusion import diffusion_generate
        print("  ✅ Generation imports OK")
    except ImportError as e:
        print(f"  ❌ Generation import error: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model can be created."""
    print("\nTesting model creation...")
    
    try:
        from jax_lm import DiffuCoderConfig, DiffuCoder
        
        # Create small config for testing
        config = DiffuCoderConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        
        # Create model
        model = DiffuCoder(config=config)
        
        # Initialize with dummy input
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        params = model.init(rng, dummy_input, deterministic=True)
        
        # Count parameters
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"  Model created with {num_params:,} parameters")
        print("  ✅ Model creation OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test model forward pass."""
    print("\nTesting forward pass...")
    
    try:
        from jax_lm import DiffuCoderConfig, DiffuCoder
        
        # Create small model
        config = DiffuCoderConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = DiffuCoder(config=config)
        
        # Initialize
        rng = jax.random.PRNGKey(0)
        batch_size = 2
        seq_len = 16
        dummy_input = jax.random.randint(
            rng, (batch_size, seq_len), 0, config.vocab_size
        )
        
        params = model.init(rng, dummy_input, deterministic=True)
        
        # Forward pass
        outputs = model.apply(params, dummy_input, deterministic=True)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, config.vocab_size + 2)  # +2 for special tokens
        assert outputs["logits"].shape == expected_shape, \
            f"Expected shape {expected_shape}, got {outputs['logits'].shape}"
        
        print(f"  Output shape: {outputs['logits'].shape}")
        print("  ✅ Forward pass OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer():
    """Test tokenizer utilities."""
    print("\nTesting tokenizer utilities...")
    
    try:
        from jax_lm.utils.tokenizer import prepare_input_ids, decode_sequences
        
        # Mock tokenizer for testing
        class MockTokenizer:
            def __init__(self):
                self.vocab = {f"token_{i}": i for i in range(1000)}
                self.pad_token_id = 0
                
            def __call__(self, texts, **kwargs):
                # Simple mock encoding
                if isinstance(texts, str):
                    texts = [texts]
                
                encoded = []
                for text in texts:
                    tokens = [hash(word) % 999 + 1 for word in text.split()]
                    encoded.append(tokens)
                
                # Pad to max length
                max_len = max(len(seq) for seq in encoded)
                if kwargs.get("padding"):
                    encoded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
                
                return {
                    "input_ids": np.array(encoded),
                    "attention_mask": np.array([[1 if t != 0 else 0 for t in seq] for seq in encoded])
                }
            
            def decode(self, tokens, **kwargs):
                return " ".join(f"token_{t}" for t in tokens if t != 0)
        
        tokenizer = MockTokenizer()
        
        # Test encoding
        texts = ["Hello world", "Testing JAX"]
        inputs = prepare_input_ids(tokenizer, texts, return_tensors="jax")
        
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape[0] == 2
        
        print("  ✅ Tokenizer utilities OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Tokenizer test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("JAX DiffuCoder Inference Test Suite")
    print("=" * 60)
    
    tests = [
        test_jax_setup,
        test_model_imports,
        test_model_creation,
        test_forward_pass,
        test_tokenizer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)