# JAX DiffuCoder Inference

Production-ready JAX/Flax implementation of DiffuCoder inference, optimized for TPU deployment with comprehensive testing and benchmarking.

## ✅ Completed Features

### 1. Core Infrastructure
- **Model Loading**: Direct pickle weight loading (no orbax dependency)
- **Proper Imports**: Fixed all model class references (Dream → DiffuCoder)
- **Tokenizer Integration**: Full HuggingFace tokenizer support
- **TPU Optimization**: Automatic device detection and optimization

### 2. Inference Pipeline
- **Standalone Script**: `jax_lm/inference.py` - Simple, self-contained inference
- **Diffusion Generation**: Full masked diffusion generation with entropy-based unmasking
- **Batch Processing**: Efficient batch inference support
- **Memory Efficiency**: bfloat16 support for TPU

### 3. Testing & Validation
- **Basic Tests**: `tests/test_inference.py` - Validates core functionality
- **Numerical Parity**: `tests/test_layer_parity.py` - Layer-wise comparison with PyTorch
- **TPU Examples**: `examples/tpu_inference_simple.py` - Ready for Google Colab TPU

## 🚀 Quick Start

### 1. Verify Installation
```bash
# Run smoke tests (no weights needed)
python jax_lm/tests/smoke_tests.py

# Expected: 10/12 tests pass (2 known issues on some platforms)
```

### 2. Local Inference
```bash
# Ensure weights are in place
# models/dream-jax/config.json (1KB)
# models/dream-jax/params.pkl (14.2GB)

# Run inference
python -m jax_lm.inference \
    --model-path ./models/dream-jax \
    --prompt "def fibonacci(n):" \
    --max-new-tokens 256

# Run quick test
python jax_lm/tests/quick_inference_test.py
```

### 3. Google Colab TPU
```python
# In Colab with TPU runtime
!pip install jax[tpu] transformers flax

# Clone repository
!git clone https://github.com/yourusername/ml-diffucoder.git
%cd ml-diffucoder

# Run TPU example
!python examples/tpu_inference_simple.py
```

### 4. Python API
```python
from jax_lm.inference import DiffuCoderInference
import jax.numpy as jnp

# Initialize
model = DiffuCoderInference(
    model_path="./models/dream-jax",
    dtype=jnp.bfloat16  # For TPU
)

# Generate
output = model.generate(
    "def quicksort(arr):",
    max_new_tokens=256,
    temperature=0.3
)
print(output)
```

## 📊 Performance Benchmarking

```python
# Run built-in benchmark
python -m jax_lm.inference --benchmark --model-path ./models/dream-jax

# Or via API
results = model.benchmark_inference(
    batch_sizes=[1, 2, 4, 8, 16],
    seq_length=128
)
```

## 🧪 Testing

```bash
# Run basic tests
python jax_lm/tests/test_inference.py

# Run numerical parity tests (requires PyTorch)
python jax_lm/tests/test_layer_parity.py \
    --pytorch-model ./models/diffucoder-7b-complete \
    --jax-model ./models/dream-jax
```

## 🔍 Numerical Correctness

The implementation includes comprehensive layer-wise parity testing:

- **Embedding Layer**: Validates token embeddings match PyTorch
- **RMSNorm Layers**: Tests normalization accuracy
- **Attention Projections**: Compares Q, K, V, O projections
- **Weight Layout**: Handles PyTorch → JAX weight transposition

Default tolerance: 1e-5 (configurable)

## 🏗️ Architecture Notes

### Key Differences from PyTorch
1. **Weight Layout**: JAX uses `[in_features, out_features]` vs PyTorch's `[out_features, in_features]`
2. **Computation**: JAX's XLA compilation provides automatic optimization
3. **Memory**: Unified memory on TPU allows efficient large model handling

### TPU Optimizations
- **bfloat16**: Native TPU support for faster computation
- **JIT Compilation**: All core functions are JIT-compiled
- **Device Sharding**: Ready for multi-TPU deployment (see `tpu_utils.py`)

## 📁 File Structure

```
jax_lm/
├── inference.py          # Main inference script
├── models/
│   └── dream.py         # Model implementation
├── generate_diffusion.py # Diffusion generation logic
├── utils/
│   ├── model_utils.py   # Model loading/saving
│   ├── tokenizer.py     # Tokenizer utilities
│   └── tpu_utils.py     # TPU-specific utilities
├── tests/
│   ├── test_inference.py      # Basic functionality tests
│   └── test_layer_parity.py   # Numerical correctness tests
└── examples/
    └── tpu_inference_simple.py # TPU example script
```

## 🚧 Known Limitations

1. **CPU Performance**: Not recommended for production (~0.005 tokens/sec)
2. **Metal Backend**: Experimental support with memory space errors
3. **Memory Requirements**: 16GB+ for inference, 32GB+ recommended
4. **Training Code**: Inference-only package (training in separate module)

## 🎯 Next Steps for Production

1. **Cloud TPU Setup**: Deploy on Cloud TPU VMs for production performance
2. **Batch Optimization**: Tune batch sizes for your specific TPU configuration
3. **Monitoring**: Add logging and metrics for production deployment
4. **API Server**: Wrap in FastAPI/Flask for HTTP inference endpoint

## 📈 Expected Performance

Based on architecture and TPU optimization:
- **TPU v3-8**: 50-100 tokens/second (estimated)
- **TPU v4-8**: 100-200 tokens/second (estimated)
- **GPU A100**: 20-40 tokens/second (estimated)
- **CPU**: <1 token/second (measured)

*Actual performance depends on batch size, sequence length, and TPU configuration.*