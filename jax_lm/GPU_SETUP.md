# DiffuCoder JAX GPU Setup Guide

This guide covers setting up and running DiffuCoder with JAX on NVIDIA GPUs.

## 🎮 GPU Requirements

### Minimum Requirements
- NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A100)
- CUDA 11.8 or higher
- cuDNN 8.6 or higher
- 32GB+ system RAM

### Recommended Setup
- NVIDIA A100 (40GB/80GB) or H100
- CUDA 12.0+
- 64GB+ system RAM
- NVMe SSD for model storage

## 🔧 Installation

### 1. Install CUDA (if not already installed)

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-3

# Add to PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

### 2. Install JAX with GPU Support

```bash
# For CUDA 12.x
pip install --upgrade "jax[cuda12]"

# For CUDA 11.x (older GPUs)
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify installation
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

### 3. Install Other Dependencies

```bash
pip install flax transformers safetensors
```

## 🚀 Running on GPU

### Basic Setup Script

```python
import os
import jax
import jax.numpy as jnp

# Force JAX to use GPU
os.environ['JAX_PLATFORMS'] = 'gpu'

# Optional: Set specific GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Verify GPU is being used
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
print(f"GPU name: {jax.devices()[0].device_kind}")

# Check GPU memory
gpu = jax.devices()[0]
print(f"GPU: {gpu}")
```

### Memory-Efficient Loading

For GPUs with limited VRAM, use memory mapping:

```python
import pickle
import numpy as np
from pathlib import Path

def load_model_gpu_efficient(model_path):
    """Load model with memory efficiency for GPU."""
    # Load config
    with open(model_path / "config.json", 'r') as f:
        config = json.load(f)
    
    # Memory-map the parameters (doesn't load into RAM immediately)
    params_path = model_path / "params.pkl"
    
    # For very large models, consider sharding
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    
    # Move to GPU progressively
    def to_gpu(x):
        if isinstance(x, dict):
            return {k: to_gpu(v) for k, v in x.items()}
        else:
            return jax.device_put(x, jax.devices('gpu')[0])
    
    # Transfer to GPU
    print("Transferring model to GPU...")
    gpu_params = to_gpu(params)
    
    return config, gpu_params
```

## ⚡ GPU Optimization Tips

### 1. Mixed Precision (bfloat16)

```python
# Enable mixed precision for better performance
from functools import partial

@partial(jax.jit, static_argnames=['training'])
def forward_mixed_precision(params, input_ids, training=False):
    # Cast parameters to bfloat16
    params_bf16 = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    
    # Run forward pass
    output = model.apply(params_bf16, input_ids, deterministic=not training)
    
    # Cast output back to float32 if needed
    return jax.tree_map(lambda x: x.astype(jnp.float32), output)
```

### 2. Gradient Checkpointing (for training)

```python
from flax.linen import remat

# Apply gradient checkpointing to save memory
@remat
def checkpointed_layer(x, params):
    # Your layer computation here
    return output
```

### 3. Efficient Batching

```python
# Optimal batch sizes for different GPUs
GPU_BATCH_SIZES = {
    'RTX 3090': 1,      # 24GB VRAM
    'RTX 4090': 2,      # 24GB VRAM
    'A100-40GB': 4,     # 40GB VRAM
    'A100-80GB': 8,     # 80GB VRAM
    'H100': 16,         # 80GB VRAM
}

def get_optimal_batch_size():
    """Determine optimal batch size based on GPU."""
    gpu_name = jax.devices()[0].device_kind
    
    # Simple heuristic based on memory
    for gpu, batch_size in GPU_BATCH_SIZES.items():
        if gpu.lower() in gpu_name.lower():
            return batch_size
    
    # Default conservative batch size
    return 1
```

## 🔍 GPU Monitoring

### Monitor GPU Usage

```python
import subprocess

def get_gpu_info():
    """Get GPU memory usage and utilization."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            used, total, util = map(int, line.split(', '))
            print(f"GPU {i}: {used}MB / {total}MB ({used/total*100:.1f}%) | Utilization: {util}%")
    except:
        print("nvidia-smi not available")

# Monitor during inference
get_gpu_info()
```

### JAX GPU Memory Profiling

```python
# Enable memory profiling
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'  # Use 90% of GPU memory

# Profile memory usage
from jax.profiler import start_trace, stop_trace

# Start profiling
start_trace("./gpu_profile")

# Your model code here
output = model.apply(params, input_ids)

# Stop profiling
stop_trace()
```

## 🐛 Common GPU Issues

### Out of Memory (OOM)

```python
# Solution 1: Reduce batch size
batch_size = 1  # Start with 1 and increase gradually

# Solution 2: Use gradient accumulation (for training)
accumulation_steps = 4

# Solution 3: Enable memory growth
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Solution 4: Clear JAX cache
jax.clear_caches()
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version
python -c "import jax; print(jax.devices())"

# If mismatch, reinstall JAX
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]"  # Match your CUDA version
```

### Multi-GPU Setup

```python
# Use multiple GPUs
devices = jax.devices('gpu')
print(f"Found {len(devices)} GPUs")

# Data parallel inference
from jax import pmap

@pmap
def parallel_forward(params, input_ids):
    return model.apply(params, input_ids, deterministic=True)

# Replicate parameters across GPUs
replicated_params = jax.device_put_replicated(params, devices)

# Run on multiple GPUs
batch_per_gpu = 2
total_batch = batch_per_gpu * len(devices)
batched_input = input_ids.reshape(len(devices), batch_per_gpu, -1)

output = parallel_forward(replicated_params, batched_input)
```

## 📊 Performance Comparison

| Hardware | Inference Time (per token) | Notes |
|----------|---------------------------|-------|
| TPU v2 | ~16-18ms | Best performance |
| A100 80GB | ~25-30ms | Good for large batches |
| A100 40GB | ~30-35ms | Limited batch size |
| RTX 4090 | ~40-45ms | Consumer GPU option |
| RTX 3090 | ~50-60ms | Minimum viable GPU |

## 🔧 Complete GPU Example

```python
#!/usr/bin/env python3
"""
Complete example for running DiffuCoder on GPU
"""

import os
import jax
import json
import pickle
from pathlib import Path

# Force GPU usage
os.environ['JAX_PLATFORMS'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

# Import model components
from jax_lm.models.dream import DreamConfig, DreamForCausalLM
from diffucoder_tokenizer import load_diffucoder_tokenizer

def main():
    # Verify GPU
    print(f"JAX backend: {jax.default_backend()}")
    devices = jax.devices()
    print(f"GPU devices: {devices}")
    
    if 'gpu' not in jax.default_backend().lower():
        raise RuntimeError("GPU not available!")
    
    # Load model
    print("\nLoading model...")
    model_path = Path("./models/dream-jax")
    
    with open(model_path / "config.json", 'r') as f:
        config = DreamConfig(**json.load(f))
    
    # Load with memory efficiency
    print("Loading parameters to GPU...")
    with open(model_path / "params.pkl", 'rb') as f:
        params = pickle.load(f)
    
    # Transfer to GPU
    params = jax.device_put(params, devices[0])
    
    # Create model and tokenizer
    model = DreamForCausalLM(config)
    tokenizer = load_diffucoder_tokenizer()
    
    # Test generation
    prompt = "def hello_world():\n    return"
    inputs = tokenizer(prompt, return_tensors="jax", padding="max_length", 
                      max_length=512, truncation=True)
    
    # JIT compile for GPU
    @jax.jit
    def generate_token(params, input_ids):
        output = model.apply(params, input_ids, deterministic=True)
        return output['logits'][:, -1, :].argmax(axis=-1)
    
    # Generate
    print(f"\nGenerating from: '{prompt}'")
    generated = inputs['input_ids']
    
    for i in range(20):
        next_token = generate_token(params, generated)
        generated = jax.numpy.concatenate([generated, next_token[:, None]], axis=1)
        
        if i < 5:
            token_text = tokenizer.decode(next_token[0])
            print(f"Token {i+1}: '{token_text}'")
    
    # Decode final result
    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated: {result}")

if __name__ == "__main__":
    main()
```

## 📚 Additional Resources

- [JAX GPU Documentation](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
- [NVIDIA GPU Optimization Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Flax GPU Best Practices](https://flax.readthedocs.io/en/latest/guides/parallel_training.html)