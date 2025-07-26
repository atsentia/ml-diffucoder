#!/usr/bin/env python3
"""PyTorch benchmark with Metal Performance Shaders (MPS) support on Mac."""

import time
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

def run_pytorch_metal_test():
    """Run PyTorch performance test with Metal acceleration."""
    if not TORCH_AVAILABLE:
        return {
            "backend": "pytorch_unavailable",
            "mean_time": 0.0,
            "std_time": 0.0,
            "tokens_per_second": 0.0,
        }
    
    print("Running PyTorch performance test with Metal acceleration...")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for Metal Performance Shaders (MPS) support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        backend_name = "pytorch_mps"
        print(f"✅ Using Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        backend_name = "pytorch_cpu"
        print(f"⚠️  MPS not available, falling back to CPU")
        # Set single thread for fair CPU comparison
        torch.set_num_threads(1)
    
    print(f"Device: {device}")
    
    # Simple transformer-like model
    batch_size = 2
    seq_len = 128
    hidden_size = 256
    num_heads = 8
    num_layers = 2
    vocab_size = 1000
    
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size + 2, hidden_size)
            
            # Transformer layers
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 2,
                    batch_first=True,
                    activation='gelu',
                    dropout=0.0,  # Disable dropout for inference
                )
                self.layers.append(layer)
            
            self.output = nn.Linear(hidden_size, vocab_size + 2)
        
        def forward(self, x):
            x = self.embeddings(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    # Create and move model to device
    model = SimpleTransformer().to(device).eval()
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e6:.2f}M")
    
    # Test data
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for i in range(5):
            output = model(input_ids)
            if device.type == "mps":
                torch.mps.synchronize()  # Ensure completion on MPS
            print(f"  Warmup {i+1}: Output shape {output.shape}")
    
    # Benchmark
    print("\nBenchmarking...")
    times = []
    with torch.no_grad():
        for i in range(10):
            if device.type == "mps":
                torch.mps.synchronize()  # Sync before timing
            
            start = time.time()
            output = model(input_ids)
            
            if device.type == "mps":
                torch.mps.synchronize()  # Wait for MPS computation
            else:
                _ = output.cpu()  # Force CPU computation
            
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    tokens_per_second = batch_size * seq_len / mean_time
    
    print(f"\nResults:")
    print(f"  Backend: {backend_name}")
    print(f"  Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
    print(f"  Throughput: {tokens_per_second:.1f} tokens/s")
    
    return {
        "backend": backend_name,
        "device": str(device),
        "model_size_mb": param_count,
        "mean_time": mean_time,
        "std_time": std_time,
        "tokens_per_second": tokens_per_second,
    }

def main():
    print("=" * 60)
    print("PyTorch Metal Performance Test")
    print("=" * 60)
    
    # Run PyTorch test
    pytorch_results = run_pytorch_metal_test()
    
    # Load JAX results for comparison if available
    try:
        with open("jax_test_results.json", "r") as f:
            jax_results = json.load(f)
        
        print(f"\n{'='*60}")
        print("Results Comparison")
        print(f"{'='*60}")
        
        print(f"\nPyTorch ({pytorch_results['backend']}):")
        print(f"  Device: {pytorch_results.get('device', 'unknown')}")
        print(f"  Mean time: {pytorch_results['mean_time']:.4f}s")
        print(f"  Throughput: {pytorch_results['tokens_per_second']:.1f} tokens/s")
        
        print(f"\nJAX CPU:")
        print(f"  Mean time: {jax_results['mean_time']:.4f}s")
        print(f"  Throughput: {jax_results['tokens_per_second']:.1f} tokens/s")
        
        # Calculate speedups
        jax_vs_pytorch = jax_results['tokens_per_second'] / pytorch_results['tokens_per_second']
        
        print(f"\nPerformance Comparison:")
        print(f"  JAX vs PyTorch: {jax_vs_pytorch:.2f}x")
        
        if pytorch_results['backend'] == 'pytorch_mps':
            print(f"\n🚀 PyTorch with Metal Performance Shaders")
            if jax_vs_pytorch > 1.0:
                improvement = (jax_vs_pytorch - 1) * 100
                print(f"   JAX is still {improvement:.1f}% faster than PyTorch+MPS")
            else:
                improvement = (1 / jax_vs_pytorch - 1) * 100
                print(f"   PyTorch+MPS is {improvement:.1f}% faster than JAX CPU!")
        else:
            print(f"\n💻 PyTorch on CPU (MPS not available)")
        
    except FileNotFoundError:
        print("\nJAX results not found for comparison")
        print("Run the JAX test first: python run_simple_jax_test.py")
    
    # Save PyTorch results
    with open("pytorch_metal_results.json", "w") as f:
        json.dump(pytorch_results, f, indent=2)
    
    print(f"\nResults saved to pytorch_metal_results.json")

if __name__ == "__main__":
    main()