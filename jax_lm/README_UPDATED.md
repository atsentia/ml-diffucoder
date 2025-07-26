# JAX/Flax DiffuCoder

A JAX/Flax implementation of DiffuCoder, optimized for TPU/GPU acceleration with full tokenizer support.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/atsentia/ml-diffucoder.git
cd ml-diffucoder

# Install JAX with TPU support (Google Colab)
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install dependencies
pip install flax transformers safetensors
```

### Download Model Weights

```bash
# Download from HuggingFace (15.2GB)
huggingface-cli download apple/DiffuCoder-7B-Instruct \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  model.safetensors.index.json \
  --local-dir models/diffucoder-7b-complete

# Download tokenizer files
huggingface-cli download apple/DiffuCoder-7B-Instruct \
  vocab.json merges.txt tokenizer_config.json tokenization_dream.py \
  --local-dir models/diffucoder-7b-complete
```

### Convert to JAX Format

```bash
python convert_dream_weights.py \
  --pytorch-model-path ./models/diffucoder-7b-complete \
  --output-path ./models/dream-jax
```

## 📖 Complete Usage Example

```python
import jax
import jax.numpy as jnp
from jax import random
from pathlib import Path
import json
import pickle

# Import DiffuCoder components
from jax_lm.models.dream import DreamConfig, DreamForCausalLM
from diffucoder_tokenizer import load_diffucoder_tokenizer

# 1. Load tokenizer
tokenizer = load_diffucoder_tokenizer("./models/diffucoder-7b-complete")

# 2. Load model configuration and weights
model_path = Path("./models/dream-jax")
with open(model_path / "config.json", 'r') as f:
    config = DreamConfig(**json.load(f))

with open(model_path / "params.pkl", 'rb') as f:
    params = pickle.load(f)

# 3. Create model
model = DreamForCausalLM(config)

# 4. Generate code
def generate_code(prompt, max_new_tokens=50):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="jax")
    input_ids = inputs['input_ids']
    
    # JIT-compiled generation
    @jax.jit
    def get_next_token(params, input_ids):
        output = model.apply(params, input_ids, deterministic=True)
        logits = output['logits'][:, -1, :]
        return jnp.argmax(logits, axis=-1)
    
    # Generate tokens
    generated = input_ids
    for _ in range(max_new_tokens):
        next_token = get_next_token(params, generated)
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)
        
        # Stop at EOS token
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    # Decode result
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example usage
prompt = "def fibonacci(n):\n    "
generated_code = generate_code(prompt)
print(generated_code)
```

## ⚡ Performance Optimization

### Fixed-Length Generation (Recommended)

To avoid JAX JIT recompilation for different sequence lengths:

```python
MAX_SEQ_LEN = 512  # Fixed sequence length

# Tokenize with padding
inputs = tokenizer(
    prompt,
    return_tensors="jax",
    padding="max_length",
    max_length=MAX_SEQ_LEN,
    truncation=True
)

# Use attention mask for proper handling
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# JIT-compiled forward pass
@jax.jit
def forward_fixed(params, input_ids, attention_mask):
    return model.apply(params, input_ids, 
                      attention_mask=attention_mask, 
                      deterministic=True)
```

## 📊 Performance Benchmarks

On Google Colab TPU v2:

| Operation | Time | Notes |
|-----------|------|-------|
| Model Loading | ~4.7s | 28GB of parameters |
| Tokenization (first) | ~200ms | Includes initialization |
| Tokenization (subsequent) | ~20-25ms | Cached |
| Raw Inference (10 tokens) | ~16ms | TPU computation |
| Per-token Generation | ~18ms | With fixed sequence length |
| JIT Compilation | ~3-7s | One-time cost |

### Benchmarking Scripts

```bash
# Basic inference benchmark
python simple_jax_benchmark.py

# Detailed timing analysis
python detailed_timing_test.py

# Optimized generation demo
python optimized_generation.py
```

## 🧪 Testing

### Run All Tests

```bash
# Tokenizer tests
python -m pytest test_tokenizer.py -v

# Model tests
python demo_outputs.py

# TPU verification
python tpu_info.py
```

### Test Real Generation

```python
# Quick test script
from diffucoder_tokenizer import load_diffucoder_tokenizer
from jax_lm.models.dream import DreamConfig, DreamForCausalLM

tokenizer = load_diffucoder_tokenizer()

# Test encoding/decoding
text = "def hello_world():"
tokens = tokenizer(text, return_tensors="jax")
decoded = tokenizer.decode(tokens['input_ids'][0])
print(f"Original: {text}")
print(f"Decoded: {decoded}")
```

## 🏗️ Model Architecture

DiffuCoder-7B specifications:
- **Parameters**: 7.6B (7,615,487,488)
- **Layers**: 28
- **Hidden Size**: 3584
- **Attention Heads**: 28
- **Key-Value Heads**: 4
- **Vocabulary Size**: 151,643
- **Max Position**: 131,072
- **Architecture**: Transformer with RoPE, RMSNorm, SwiGLU

## 🛠️ Advanced Features

### Multi-Device Sharding

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P

# Create device mesh
devices = mesh_utils.create_device_mesh((len(jax.devices()),))
mesh = Mesh(devices, axis_names=('batch',))

# Shard model across devices
with mesh:
    # Your model code here
    pass
```

### Custom Generation Strategies

```python
def generate_with_sampling(prompt, temperature=0.7, top_k=50, top_p=0.95):
    """Generate with temperature, top-k, and nucleus sampling."""
    # See optimized_generation.py for full implementation
    pass
```

## 📁 Project Structure

```
jax_lm/
├── models/
│   ├── __init__.py
│   ├── dream.py          # Core model implementation
│   └── components.py     # Model components
├── training/
│   └── coupled_grpo.py   # Training implementation
├── utils/
│   └── tokenizer.py      # Tokenizer utilities
├── tests/
│   └── test_numerical_parity.py
├── README.md            # This file
├── TOKENIZER.md        # Tokenizer documentation
└── BENCHMARKS.md       # Performance analysis
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest jax_lm/tests/`
4. Submit a pull request

## 📄 License

Apache License 2.0 - See [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

- Apple ML Research for the original DiffuCoder
- JAX and Flax teams for excellent frameworks
- HuggingFace for model hosting and tools

## 📞 Support

- Issues: [GitHub Issues](https://github.com/atsentia/ml-diffucoder/issues)
- Discussions: [GitHub Discussions](https://github.com/atsentia/ml-diffucoder/discussions)