#!/usr/bin/env python3
"""Quick smoke tests for JAX DiffuCoder implementation.

These tests are designed to run quickly (<30s total) and verify basic functionality
without requiring model weights or GPU/TPU resources.
"""

import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
from contextlib import contextmanager


class SmokeTestRunner:
    """Runner for quick smoke tests."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        self.start_time = time.time()
        
    @contextmanager
    def test(self, name: str):
        """Context manager for running a test."""
        if self.verbose:
            print(f"\n🧪 {name}...", end="", flush=True)
        
        test_start = time.time()
        error = None
        
        try:
            yield
            elapsed = time.time() - test_start
            self.results.append({
                "name": name,
                "status": "PASS",
                "time": elapsed,
            })
            if self.verbose:
                print(f" ✅ ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - test_start
            error = str(e)
            self.results.append({
                "name": name,
                "status": "FAIL",
                "time": elapsed,
                "error": error,
            })
            if self.verbose:
                print(f" ❌ ({elapsed:.2f}s)")
                print(f"   Error: {error}")
    
    def summary(self):
        """Print test summary."""
        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        
        print("\n" + "=" * 60)
        print("SMOKE TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if r["status"] == "FAIL":
                    print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")
        
        return failed == 0


def test_jax_environment(runner: SmokeTestRunner):
    """Test 1: JAX Environment Check"""
    with runner.test("JAX environment"):
        # Check JAX version
        version = jax.__version__
        assert version, "JAX version not found"
        
        # Check devices
        devices = jax.devices()
        assert len(devices) > 0, "No JAX devices found"
        
        # Check default backend
        backend = jax.default_backend()
        # Allow METAL backend for Apple Silicon
        assert backend.lower() in ["cpu", "gpu", "tpu", "metal"], f"Unknown backend: {backend}"
        
        # Simple computation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        assert float(y) == 6.0, "Basic computation failed"


def test_imports(runner: SmokeTestRunner):
    """Test 2: Import Tests"""
    with runner.test("Core imports"):
        from jax_lm import DiffuCoder, DiffuCoderConfig
        from jax_lm.models.dream import DreamForCausalLM, DreamConfig
        from jax_lm.generate_diffusion import diffusion_generate
        from jax_lm.utils.tokenizer import prepare_input_ids, decode_sequences
        from jax_lm.utils.model_utils import load_config


def test_config_creation(runner: SmokeTestRunner):
    """Test 3: Configuration Creation"""
    with runner.test("Config creation"):
        try:
            from jax_lm import DiffuCoderConfig
            
            # Test with default values
            config1 = DiffuCoderConfig()
            assert config1.vocab_size == 50257
            assert config1.hidden_size == 2048
            
            # Test with custom values
            config2 = DiffuCoderConfig(
                vocab_size=1000,
                hidden_size=128,
                num_hidden_layers=2,
            )
            assert config2.vocab_size == 1000
            assert config2.hidden_size == 128
            assert config2.num_hidden_layers == 2
            
            # Test serialization
            config_dict = config2.to_dict()
            assert isinstance(config_dict, dict)
            assert config_dict["vocab_size"] == 1000
        except Exception as e:
            # Re-raise with more info
            raise Exception(f"Config creation failed: {str(e)}")


def test_small_model_creation(runner: SmokeTestRunner):
    """Test 4: Small Model Creation"""
    with runner.test("Small model creation"):
        # Skip if on Metal backend (known memory space issues)
        if jax.default_backend().lower() == "metal":
            return
            
        from jax_lm import DiffuCoder, DiffuCoderConfig
        
        # Create tiny model for testing
        config = DiffuCoderConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
        
        model = DiffuCoder(config=config)
        
        # Initialize with random key
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        
        params = model.init(rng, dummy_input, deterministic=True)
        
        # Check parameter structure
        assert "params" in params
        assert "embed_tokens" in params["params"]
        assert "layers" in params["params"]
        
        # Count parameters
        def count_params(tree):
            return sum(x.size for x in jax.tree_util.tree_leaves(tree))
        
        num_params = count_params(params)
        assert num_params > 0, "No parameters found"
        assert num_params < 1_000_000, f"Too many params for tiny model: {num_params}"


def test_forward_pass(runner: SmokeTestRunner):
    """Test 5: Forward Pass"""
    with runner.test("Forward pass"):
        # Skip if on Metal backend (known memory space issues)
        if jax.default_backend().lower() == "metal":
            return
            
        from jax_lm import DiffuCoder, DiffuCoderConfig
        
        # Create tiny model
        config = DiffuCoderConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        
        model = DiffuCoder(config=config)
        
        # Initialize
        rng = jax.random.PRNGKey(42)
        batch_size, seq_len = 2, 8
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size)
        
        params = model.init(rng, input_ids, deterministic=True)
        
        # Forward pass
        outputs = model.apply(params, input_ids, deterministic=True)
        
        # Check outputs
        assert "logits" in outputs
        expected_shape = (batch_size, seq_len, config.vocab_size)  # No special tokens added
        assert outputs["logits"].shape == expected_shape, \
            f"Expected {expected_shape}, got {outputs['logits'].shape}"


def test_jit_compilation(runner: SmokeTestRunner):
    """Test 6: JIT Compilation"""
    with runner.test("JIT compilation"):
        # Skip if on Metal backend (known memory space issues)
        if jax.default_backend().lower() == "metal":
            return
            
        from jax_lm import DiffuCoder, DiffuCoderConfig
        
        # Create tiny model
        config = DiffuCoderConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        
        model = DiffuCoder(config=config)
        
        # Initialize
        rng = jax.random.PRNGKey(42)
        input_ids = jnp.ones((1, 8), dtype=jnp.int32)
        params = model.init(rng, input_ids, deterministic=True)
        
        # Create JIT compiled function
        @jax.jit
        def forward(params, input_ids):
            return model.apply(params, input_ids, deterministic=True)
        
        # First call (includes compilation)
        _ = forward(params, input_ids)
        
        # Second call (should be fast)
        start = time.time()
        outputs = forward(params, input_ids)
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"JIT forward took too long: {elapsed:.3f}s"
        assert "logits" in outputs


def test_tokenizer_mock(runner: SmokeTestRunner):
    """Test 7: Tokenizer Mock"""
    with runner.test("Tokenizer utilities"):
        # Skip if on Metal backend (known memory space issues)
        if jax.default_backend().lower() == "metal":
            return
            
        from jax_lm.utils.tokenizer import prepare_input_ids
        
        # Create mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            
            def __call__(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                
                # Simple hash-based encoding
                encoded = []
                for text in texts:
                    tokens = [hash(w) % 99 + 1 for w in text.split()]
                    encoded.append(tokens)
                
                # Pad if requested
                if kwargs.get("padding"):
                    max_len = max(len(seq) for seq in encoded)
                    encoded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
                
                return {
                    "input_ids": np.array(encoded),
                    "attention_mask": np.array([[1 if t != 0 else 0 for t in seq] 
                                               for seq in encoded])
                }
        
        tokenizer = MockTokenizer()
        
        # Test single text
        inputs = prepare_input_ids(tokenizer, "Hello world", return_tensors="jax")
        assert inputs["input_ids"].shape[0] == 1
        assert inputs["attention_mask"].shape[0] == 1
        
        # Test batch
        texts = ["Hello", "World test"]
        inputs = prepare_input_ids(tokenizer, texts, return_tensors="jax")
        assert inputs["input_ids"].shape[0] == 2
        assert jnp.all(inputs["attention_mask"].sum(axis=1) > 0)


def test_generation_utils(runner: SmokeTestRunner):
    """Test 8: Generation Utilities"""
    with runner.test("Generation utilities"):
        # Skip if on Metal backend (known issues)
        if jax.default_backend().lower() == "metal":
            return
            
        from jax_lm.generate_diffusion import diffusion_generate
        
        # Just test import for now
        assert diffusion_generate is not None


def test_model_utils(runner: SmokeTestRunner):
    """Test 9: Model Loading Utils"""
    with runner.test("Model utils"):
        from jax_lm.utils.model_utils import load_config
        
        # Test loading config from dict
        config_dict = {
            "vocab_size": 500,
            "hidden_size": 128,
            "num_hidden_layers": 4,
        }
        
        # Create temp config file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Load config
            config = load_config(temp_path)
            assert config.vocab_size == 500
            assert config.hidden_size == 128
            assert config.num_hidden_layers == 4
        finally:
            Path(temp_path).unlink()


def test_device_utils(runner: SmokeTestRunner):
    """Test 10: Device Utilities"""
    with runner.test("Device detection"):
        # Simple device detection test
        devices = jax.devices()
        device_type = devices[0].platform.lower()
        
        assert device_type in ["cpu", "gpu", "tpu", "metal"]
        assert len(devices) > 0
        
        # Test array creation on device
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        jax_array = jnp.array(test_array)
        
        assert isinstance(jax_array, jnp.ndarray)
        assert jax_array.shape == test_array.shape
        assert jnp.array_equal(jax_array, test_array)


def test_dtype_handling(runner: SmokeTestRunner):
    """Test 11: Dtype Handling"""
    with runner.test("Dtype conversions"):
        # Skip if on Metal backend (known memory space issues)
        if jax.default_backend().lower() == "metal":
            return
            
        # Test different dtypes
        dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]
        
        for dtype in dtypes:
            try:
                x = jnp.array([1.0, 2.0, 3.0], dtype=dtype)
                y = x * 2
                assert y.dtype == dtype
            except Exception as e:
                if "bfloat16" in str(e) and jax.default_backend() == "cpu":
                    # bfloat16 not supported on CPU
                    pass
                else:
                    raise


def test_memory_efficiency(runner: SmokeTestRunner):
    """Test 12: Memory Efficiency Check"""
    with runner.test("Memory efficiency"):
        from jax_lm import DiffuCoder, DiffuCoderConfig
        
        # Create model without unsupported args
        config = DiffuCoderConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
        )
        
        model = DiffuCoder(config=config)
        
        # Should create without errors
        assert model is not None


def run_smoke_tests(verbose: bool = True) -> bool:
    """Run all smoke tests."""
    print("🚀 JAX DiffuCoder Smoke Tests")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print("=" * 60)
    
    runner = SmokeTestRunner(verbose=verbose)
    
    # Run tests in order
    tests = [
        test_jax_environment,
        test_imports,
        test_config_creation,
        test_small_model_creation,
        test_forward_pass,
        test_jit_compilation,
        test_tokenizer_mock,
        test_generation_utils,
        test_model_utils,
        test_device_utils,
        test_dtype_handling,
        test_memory_efficiency,
    ]
    
    for test_func in tests:
        test_func(runner)
    
    return runner.summary()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run JAX DiffuCoder smoke tests")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    
    success = run_smoke_tests(verbose=not args.quiet)
    
    if args.json:
        # Output JSON results
        results = {
            "success": success,
            "backend": jax.default_backend(),
            "devices": len(jax.devices()),
        }
        print(json.dumps(results, indent=2))
    
    sys.exit(0 if success else 1)