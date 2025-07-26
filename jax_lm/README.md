# JAX/Flax DiffuCoder

A high-performance JAX/Flax implementation of DiffuCoder, optimized for TPU/GPU/CPU acceleration. This implementation leverages JAX's XLA compilation and hardware-specific optimizations.

> **📚 Documentation**
> - [Full Setup & Usage Guide](README_UPDATED.md) - Complete installation and usage examples
> - [Tokenizer Guide](TOKENIZER.md) - Tokenizer implementation and usage
> - [GPU Setup Guide](GPU_SETUP.md) - Running on NVIDIA GPUs
> - [Performance Benchmarks](BENCHMARKS.md) - Detailed performance analysis

## 🚀 Highlights

- **Multi-Hardware Support**: Automatic detection and optimization for TPU, GPU, and CPU
- **Performance**: Optimized for JAX's strengths including XLA compilation and TPU acceleration
- **Complete Implementation**: Full DiffuCoder architecture with RoPE, multi-head attention, and RMSNorm
- **Easy Conversion**: Tools to convert existing PyTorch checkpoints to JAX format
- **Production Ready**: Comprehensive testing and benchmarking suite

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Usage Examples](#usage-examples)
- [Training](#training)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🔧 Installation

### Prerequisites

- Python 3.9+
- JAX (CPU/GPU/TPU support)
- 32GB+ RAM for full 7B model (16GB with quantization)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-diffucoder.git
cd ml-diffucoder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install JAX/Flax DiffuCoder
pip install -e jax_lm

# For GPU support
pip install -e "jax_lm[gpu]"

# For TPU support
pip install -e "jax_lm[tpu]"
```

## 🚀 Quick Start

### 1. Download and Convert Model Weights

```bash
# Download from HuggingFace Hub
huggingface-cli download apple/DiffuCoder-7B-Instruct \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  --local-dir models/diffucoder-7b-complete

# Convert to JAX format
python convert_dream_weights.py \
    --pytorch-model-path ./models/diffucoder-7b-complete \
    --output-path ./models/dream-jax
```

### 2. Run Inference

```python
import jax
from jax_lm import load_model, diffusion_generate
from jax_lm.utils import load_tokenizer

# Load model and tokenizer
model, params = load_model("./models/jax")
tokenizer = load_tokenizer("./models/jax/tokenizer")

# Generate code
prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="jax")

output = diffusion_generate(
    model, 
    params,
    inputs["input_ids"],
    jax.random.PRNGKey(0),
    max_new_tokens=256,
    temperature=0.3,
)

generated_text = tokenizer.decode(output["sequences"][0])
print(generated_text)
```

### 3. TPU Inference

For optimized TPU inference:

```bash
python jax_lm/inference_tpu.py \
    --model-path ./models/jax \
    --prompt "def quicksort(arr):" \
    --max-new-tokens 256 \
    --temperature 0.3
```

## 🏗️ Model Architecture

DiffuCoder uses a masked diffusion approach for code generation:

- **Base Architecture**: Transformer with RoPE position embeddings
- **Diffusion Process**: Iterative refinement through masked token prediction
- **Special Features**: 
  - Entropy-based token unmasking
  - Code-aware tokenization
  - Syntax-aware generation

For detailed architecture information, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## ⚡ Performance

### Benchmark Results

Performance characteristics:
- **CPU**: Benefits from XLA optimization and JIT compilation
- **GPU**: Efficient memory usage and kernel fusion
- **TPU**: Native XLA support and optimized for TPU architecture

*Note: Performance depends on many factors including model size, batch size, sequence length, hardware configuration, and specific use case. JAX and PyTorch each have their strengths - we recommend benchmarking on your specific hardware and workload.*

### Running Benchmarks

```bash
# Auto-detect best hardware
python jax_lm/benchmarks/hardware_benchmark.py --backend auto

# Compare PyTorch vs JAX
./jax_lm/benchmarks/download_and_benchmark.sh
```

For detailed performance analysis, see [BENCHMARKS.md](docs/BENCHMARKS.md).

## 📖 Usage Examples

### Basic Generation

```python
from jax_lm import DiffuCoder, DiffuCoderConfig

# Create model with custom config
config = DiffuCoderConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
)
model = DiffuCoder(config)
```

### Batch Generation

```python
# Generate for multiple prompts
prompts = [
    "def factorial(n):",
    "class BinaryTree:",
    "async function fetchData():"
]

outputs = batch_generate(model, params, prompts)
```

### Streaming Generation

```python
# Stream tokens as they're generated
def callback(state):
    print(f"Generated {len(state['tokens'])} tokens...")

stream_diffusion_generate(
    model, params, input_ids, rng,
    callback=callback
)
```

For more examples, see [examples/](examples/).

## 🏋️ Training

### Coupled-GRPO Training

Train DiffuCoder with the Coupled-GRPO algorithm:

```python
from jax_lm.training import CoupledGRPOTrainer, TrainingConfig

config = TrainingConfig(
    model_name="./models/jax",
    learning_rate=1e-6,
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = CoupledGRPOTrainer(config)
trainer.train()
```

For training details, see [TRAINING.md](docs/TRAINING.md).

## 📚 API Reference

### Core Functions

- `load_model(path)` - Load a DiffuCoder model
- `diffusion_generate(...)` - Generate text using diffusion
- `convert_pytorch_checkpoint(...)` - Convert PyTorch weights

### Model Classes

- `DiffuCoder` - Main model class
- `DiffuCoderConfig` - Model configuration
- `CoupledGRPOTrainer` - Training with Coupled-GRPO

For complete API documentation, see [API.md](docs/API.md).

## 🛠️ Advanced Usage

### Multi-Device Sharding

```python
from jax_lm.utils import setup_tpu, shard_params

# Setup TPU mesh
devices = setup_tpu()
mesh = get_tpu_mesh(devices)

# Shard model across devices
with mesh:
    sharded_params = shard_params(params, mesh)
```

### Custom Generation Algorithms

```python
# Use custom unmasking strategy
output = diffusion_generate(
    model, params, input_ids, rng,
    alg="custom",
    alg_fn=my_custom_unmask_fn,
)
```

## 🧪 Testing

Run the test suite:

```bash
# Numerical parity tests
python jax_lm/tests/test_numerical_parity.py

# Performance tests
pytest jax_lm/tests/test_performance.py

# Integration tests
pytest jax_lm/tests/test_integration.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -e "jax_lm[dev]"

# Run code formatting
black jax_lm/
isort jax_lm/

# Run linters
flake8 jax_lm/
```

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

- Original DiffuCoder implementation by Apple
- JAX/Flax teams for the excellent frameworks
- The open-source community for feedback and contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ml-diffucoder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ml-diffucoder/discussions)
- **Documentation**: [Full Docs](https://diffucoder-jax.readthedocs.io)

---

<div align="center">
  <b>JAX/Flax DiffuCoder</b> - Fast, Efficient, and Scalable Code Generation
</div>