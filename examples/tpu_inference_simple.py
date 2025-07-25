#!/usr/bin/env python3
"""Simple TPU-ready inference example for DiffuCoder.

This script can be run on Google Colab TPU or Cloud TPU VM.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random

# Check if running on Colab
IN_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ

if IN_COLAB:
    print("🔧 Setting up Colab environment...")
    # Install required packages
    os.system("pip install -q transformers sentencepiece")


def setup_tpu():
    """Setup TPU if available."""
    try:
        # Check for TPU
        if 'COLAB_TPU_ADDR' in os.environ:
            import jax.tools.colab_tpu
            jax.tools.colab_tpu.setup_tpu()
            print("✅ Colab TPU initialized")
        
        devices = jax.devices()
        print(f"Available devices: {devices}")
        print(f"Device count: {len(devices)}")
        print(f"Device type: {devices[0].platform}")
        
        return devices
    except Exception as e:
        print(f"⚠️  TPU setup failed: {e}")
        print("Falling back to CPU/GPU")
        return jax.devices()


def download_model_if_needed():
    """Download model weights if not present (for Colab)."""
    model_path = Path("./models/dream-jax")
    
    if not model_path.exists():
        print("📥 Model not found locally. Please upload or mount the model files.")
        print("Expected structure:")
        print("  models/dream-jax/")
        print("    ├── config.json")
        print("    └── params.pkl")
        return None
    
    return model_path


def simple_inference_demo():
    """Run a simple inference demo."""
    print("\n🚀 DiffuCoder JAX Inference Demo")
    print("=" * 60)
    
    # Setup devices
    devices = setup_tpu()
    
    # Model path
    model_path = download_model_if_needed()
    if model_path is None:
        return
    
    # Import after setup
    from jax_lm.inference import DiffuCoderInference
    
    # Initialize model
    print("\n📦 Loading model...")
    model = DiffuCoderInference(
        model_path=model_path,
        dtype=jnp.bfloat16,  # Use bfloat16 for TPU
    )
    
    # Test prompts
    prompts = [
        "def fibonacci(n):",
        "class BinaryTree:",
        "def quicksort(arr):",
    ]
    
    print("\n🎯 Running inference...")
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Generate
        start_time = time.time()
        output = model.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.3,
            num_steps=50,  # Fewer steps for demo
        )
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"📄 Generated:\n{output}")
        print("-" * 40)
    
    # Benchmark if on TPU
    if devices[0].platform.upper() == "TPU":
        print("\n📊 Running TPU benchmark...")
        results = model.benchmark_inference(
            batch_sizes=[1, 2, 4, 8, 16],
            seq_length=128,
            num_iterations=10,
        )
        
        print("\nBenchmark Results:")
        for batch_size, metrics in results.items():
            print(f"  {batch_size}: {metrics['throughput_tokens_per_second']:.0f} tokens/s")


def colab_snippet():
    """Print Colab setup snippet."""
    print("""
To run this on Google Colab TPU:

1. Create a new Colab notebook
2. Go to Runtime → Change runtime type → TPU
3. Run these cells:

```python
# Cell 1: Mount Drive (if model is in Drive)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install packages
!pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install -q transformers sentencepiece flax

# Cell 3: Clone repo and run
!git clone https://github.com/yourusername/ml-diffucoder.git
%cd ml-diffucoder
!python examples/tpu_inference_simple.py
```
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--colab-setup", action="store_true", help="Show Colab setup instructions")
    args = parser.parse_args()
    
    if args.colab_setup:
        colab_snippet()
    else:
        simple_inference_demo()