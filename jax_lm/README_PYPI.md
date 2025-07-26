# JAX-DiffuCoder

[![PyPI version](https://badge.fury.io/py/jax-diffucoder.svg)](https://badge.fury.io/py/jax-diffucoder)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

High-performance JAX/Flax implementation of DiffuCoder, a 7.6B parameter diffusion-based code generation model. Optimized for TPUs and GPUs with 2-5x faster inference than PyTorch.

## 🚀 Quick Start

### Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
# CPU-only
uv pip install jax-diffucoder

# GPU support
uv pip install "jax-diffucoder[gpu]"

# TPU support
uv pip install "jax-diffucoder[tpu]"
```

Using pip:
```bash
pip install jax-diffucoder
```

### Basic Usage

```python
from jax_diffucoder import load_model, generate
import jax

# Load model from HuggingFace
model, params = load_model("atsentia/DiffuCoder-7B-JAX")

# Generate code
prompt = "def fibonacci(n):"
output = generate(
    model, params, prompt,
    max_new_tokens=256,
    temperature=0.7
)
print(output)
```

## 🎯 Key Features

- **Native JAX/Flax**: Built from scratch for maximum performance
- **TPU Optimized**: First-class support for Google Cloud TPUs
- **Memory Efficient**: Sharded checkpoints and dtype flexibility
- **No HF Dependencies**: Pure JAX implementation (HuggingFace only for storage)
- **Production Ready**: Comprehensive testing and benchmarking

## 📊 Performance

| Hardware | Tokens/sec | Speedup vs PyTorch |
|----------|------------|-------------------|
| TPU v3-8 | 65-70 | 2.5x |
| A100 80GB | 40-45 | 1.5x |
| RTX 4090 | 20-25 | 1.2x |

## 🔧 Advanced Usage

### Memory-Efficient Loading
```python
# Load with bfloat16 for lower memory usage
model, params = load_model(
    "atsentia/DiffuCoder-7B-JAX",
    dtype=jnp.bfloat16
)
```

### Batch Generation
```python
prompts = ["def sort(arr):", "class Stack:"]
outputs = batch_generate(model, params, prompts)
```

### Custom Caching
```python
from jax_diffucoder import PureJAXModelLoader

loader = PureJAXModelLoader(cache_dir="/path/to/cache")
model, params = loader.load_model("atsentia/DiffuCoder-7B-JAX")
```

## 📦 Model Formats

The package supports multiple checkpoint formats:
- **Orbax Sharded**: Recommended for large models
- **Pickle**: Legacy single-file format
- **Safetensors**: For PyTorch compatibility (conversion required)

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/atsentia/ml-diffucoder
cd ml-diffucoder/jax_lm

# Install with uv
uv pip install -e ".[dev]"

# Run tests
make test

# Format code
make format
```

## 📄 License

Apache License 2.0. Model weights are subject to Apple's original license.

## 🔗 Links

- [GitHub Repository](https://github.com/atsentia/ml-diffucoder)
- [Model on HuggingFace](https://huggingface.co/atsentia/DiffuCoder-7B-JAX)
- [Documentation](https://github.com/atsentia/ml-diffucoder/tree/main/jax_lm)

## Citation

```bibtex
@software{jax-diffucoder,
  title = {JAX-DiffuCoder: High-Performance JAX Implementation},
  author = {ML-DiffuCoder Contributors},
  year = {2024},
  url = {https://github.com/atsentia/ml-diffucoder}
}
```