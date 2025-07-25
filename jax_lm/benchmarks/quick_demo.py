#!/usr/bin/env python3
"""Quick demo benchmark comparing PyTorch and JAX implementations."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import torch
import torch.nn as nn

# Force CPU for fair comparison
jax.config.update('jax_platform_name', 'cpu')
torch.set_num_threads(1)  # Single thread for fair comparison


def create_mini_pytorch_model(hidden_size=256, num_layers=4):
    """Create a mini PyTorch transformer for testing."""
    class MiniTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_size)
            self.output = nn.Linear(hidden_size, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.output(x)
    
    return MiniTransformer().eval()


def create_mini_jax_model(hidden_size=256, num_layers=4):
    """Create equivalent JAX model."""
    import flax.linen as nn
    
    class MiniTransformer(nn.Module):
        hidden_size: int = hidden_size
        num_layers: int = num_layers
        
        @nn.compact
        def __call__(self, x):
            x = nn.Embed(num_embeddings=1000, features=self.hidden_size)(x)
            
            # Simplified transformer layers
            for i in range(self.num_layers):
                # Self-attention
                attn = nn.MultiHeadDotProductAttention(
                    num_heads=8,
                    qkv_features=self.hidden_size,
                    out_features=self.hidden_size,
                )(x)
                x = nn.LayerNorm()(x + attn)
                
                # FFN
                ffn = nn.Dense(self.hidden_size * 4)(x)
                ffn = nn.relu(ffn)
                ffn = nn.Dense(self.hidden_size)(ffn)
                x = nn.LayerNorm()(x + ffn)
            
            x = nn.Dense(1000)(x)
            return x
    
    return MiniTransformer()


def benchmark_forward_pass():
    """Compare forward pass performance."""
    print("=" * 60)
    print("Mini Benchmark: PyTorch vs JAX Forward Pass")
    print("=" * 60)
    
    # Model parameters
    hidden_size = 256
    num_layers = 4
    batch_size = 2
    seq_len = 128
    num_runs = 10
    
    # Create models
    print(f"Creating models (hidden_size={hidden_size}, layers={num_layers})...")
    pytorch_model = create_mini_pytorch_model(hidden_size, num_layers)
    jax_model = create_mini_jax_model(hidden_size, num_layers)
    
    # Initialize JAX model
    rng = random.PRNGKey(0)
    dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
    params = jax_model.init(rng, dummy_input)
    
    # Count parameters
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    jax_params = sum(x.size for x in jax.tree_leaves(params))
    print(f"PyTorch parameters: {pytorch_params:,}")
    print(f"JAX parameters: {jax_params:,}")
    
    # Create inputs
    input_ids_np = np.random.randint(0, 1000, (batch_size, seq_len))
    pytorch_input = torch.tensor(input_ids_np, dtype=torch.long)
    jax_input = jnp.array(input_ids_np, dtype=jnp.int32)
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = pytorch_model(pytorch_input)
    _ = jax_model.apply(params, jax_input)
    
    # Benchmark PyTorch
    print(f"\nBenchmarking PyTorch (batch_size={batch_size}, seq_len={seq_len})...")
    pytorch_times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            output = pytorch_model(pytorch_input)
            _ = output.numpy()  # Force computation
            elapsed = time.time() - start
            pytorch_times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.4f}s")
    
    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    
    # JIT compile JAX
    jax_forward = jax.jit(jax_model.apply)
    
    # Benchmark JAX
    print(f"\nBenchmarking JAX (batch_size={batch_size}, seq_len={seq_len})...")
    jax_times = []
    for i in range(num_runs):
        start = time.time()
        output = jax_forward(params, jax_input)
        output.block_until_ready()  # Force computation
        elapsed = time.time() - start
        jax_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    jax_mean = np.mean(jax_times)
    jax_std = np.std(jax_times)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"PyTorch: {pytorch_mean:.4f}s ± {pytorch_std:.4f}s")
    print(f"JAX:     {jax_mean:.4f}s ± {jax_std:.4f}s")
    print(f"Speedup: {pytorch_mean/jax_mean:.2f}x")
    
    tokens_per_sec_pytorch = batch_size * seq_len / pytorch_mean
    tokens_per_sec_jax = batch_size * seq_len / jax_mean
    print(f"\nThroughput:")
    print(f"PyTorch: {tokens_per_sec_pytorch:.1f} tokens/s")
    print(f"JAX:     {tokens_per_sec_jax:.1f} tokens/s")
    
    # Test numerical accuracy
    print("\n" + "=" * 60)
    print("NUMERICAL COMPARISON")
    print("=" * 60)
    
    # Simple forward pass comparison
    with torch.no_grad():
        pytorch_out = pytorch_model(pytorch_input[:1, :32]).numpy()
    jax_out = jax_model.apply(params, jax_input[:1, :32])
    
    # Compare outputs
    abs_diff = np.abs(pytorch_out - jax_out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Output shape: {pytorch_out.shape}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    
    if max_diff < 0.01:
        print("Numerical accuracy: GOOD ✓")
    else:
        print("Numerical accuracy: Differences detected ⚠️")
    
    return {
        "pytorch_mean": pytorch_mean,
        "jax_mean": jax_mean,
        "speedup": pytorch_mean / jax_mean,
        "max_diff": max_diff,
    }


def test_jax_compilation():
    """Test JAX compilation and platform detection."""
    print("\n" + "=" * 60)
    print("JAX Platform Information")
    print("=" * 60)
    
    # Platform info
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Test computation
    x = jnp.ones((100, 100))
    y = jnp.dot(x, x)
    print(f"Test computation successful: {y.shape}")
    print(f"Computation device: {y.device()}")
    
    # Test JIT compilation
    @jax.jit
    def test_fn(x):
        return jnp.sum(x ** 2)
    
    result = test_fn(x)
    print(f"JIT compilation successful: {result}")


def main():
    """Run the demo benchmark."""
    print("DiffuCoder JAX Implementation Demo")
    print("This is a simplified benchmark for demonstration")
    print("")
    
    # Test JAX setup
    test_jax_compilation()
    
    # Run benchmark
    results = benchmark_forward_pass()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results["speedup"] > 1.0:
        print(f"JAX is {results['speedup']:.2f}x faster than PyTorch! 🚀")
    else:
        print(f"PyTorch is {1/results['speedup']:.2f}x faster than JAX")
    
    print("\nNote: This is a simplified demo. Full benchmarks would include:")
    print("- Actual DiffuCoder model architecture")
    print("- Diffusion generation benchmarks")
    print("- Memory usage comparison")
    print("- Multi-device scaling")
    print("\nTo run full benchmarks, use:")
    print("  ./jax_lm/benchmarks/download_and_benchmark.sh")


if __name__ == "__main__":
    main()