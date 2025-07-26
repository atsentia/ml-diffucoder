# JAX-DiffuCoder Development Guide

## 🚀 Quick Setup with UV

We use [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Development Setup
```bash
# Clone repository
git clone https://github.com/atsentia/ml-diffucoder
cd ml-diffucoder/jax_lm

# Install development environment
make install-dev

# Or manually with uv
uv pip install -e ".[dev,tokenizer]"
```

## 🏗️ Project Structure

```
jax_lm/
├── jax_lm/
│   ├── __init__.py          # Package exports
│   ├── models/              # Model implementations
│   │   ├── diffucoder.py    # Main model class
│   │   └── components.py    # Model components
│   ├── utils/               # Utilities
│   │   ├── pure_jax_loader.py    # HF-independent loader
│   │   ├── orbax_sharding.py     # Sharded checkpoints
│   │   └── tokenizer.py          # Tokenizer implementation
│   ├── training/            # Training utilities
│   └── benchmarks/          # Performance benchmarks
├── tests/                   # Test suite
├── examples/                # Usage examples
├── pyproject.toml          # Package configuration
├── uv.toml                 # UV configuration
└── Makefile                # Development commands
```

## 🧪 Testing

### Run All Tests
```bash
make test
```

### Specific Test Categories
```bash
# Model loading tests
make test-loading

# Integration tests
make test-integration

# Fast tests only (no slow tests)
make test-fast
```

### Writing Tests
```python
# tests/test_new_feature.py
import unittest
from jax_lm import DiffuCoder

class TestNewFeature(unittest.TestCase):
    def test_feature(self):
        # Test implementation
        pass
```

## 🎨 Code Style

We use Black, isort, and flake8 for consistent code style.

### Format Code
```bash
make format
```

### Check Linting
```bash
make lint
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📦 Building and Publishing

### Build Package
```bash
make build
```

### Test PyPI Upload
```bash
make upload-test
```

### Production PyPI Upload
```bash
make upload
```

## 🔄 HuggingFace Integration

### Upload Model to HuggingFace
```bash
export HF_TOKEN=your_token
export REPO_ID=username/model-name

make hf-upload
```

### Test HuggingFace Loading
```bash
make hf-test-load
```

## 🐛 Debugging

### JAX Device Information
```python
import jax
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

### Memory Profiling
```python
from jax.profiler import device_memory_profile
print(device_memory_profile())
```

### Enable JAX Debugging
```bash
export JAX_ENABLE_CHECKS=1
export JAX_DISABLE_JIT=1
python your_script.py
```

## 📊 Benchmarking

### Run Benchmarks
```bash
# CPU benchmark
make benchmark-cpu

# GPU benchmark
make benchmark-gpu

# TPU benchmark
make benchmark-tpu
```

### Adding New Benchmarks
```python
# jax_lm/benchmarks/new_benchmark.py
import time
from jax_lm import load_model

def benchmark_feature():
    model, params = load_model("atsentia/DiffuCoder-7B-JAX")
    
    start = time.time()
    # Benchmark code here
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
```

## 🔧 Common Issues

### UV Installation Issues
```bash
# If uv is not in PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### JAX GPU Issues
```bash
# Check CUDA version
nvidia-smi

# Install specific CUDA version
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Memory Issues
```python
# Use bfloat16 to reduce memory
model, params = load_model("model", dtype=jnp.bfloat16)

# Enable gradient checkpointing
config = DiffuCoderConfig(gradient_checkpointing=True)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Format code (`make format`)
6. Commit (`git commit -m 'Add amazing feature'`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 Documentation

### Generate API Docs
```bash
make docs
```

### Writing Docstrings
```python
def function(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> function(42, "test")
        True
    """
    pass
```

## 🔍 Useful Commands

### Clean Build Artifacts
```bash
make clean
```

### Full Development Check
```bash
make pre-commit
```

### Setup New Development Environment
```bash
make setup-dev
```

## 📚 Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Orbax Documentation](https://orbax.readthedocs.io/)
- [UV Documentation](https://github.com/astral-sh/uv)