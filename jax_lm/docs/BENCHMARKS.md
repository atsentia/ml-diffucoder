# Performance Benchmarks

This document describes how to run benchmarks and interpret results for JAX DiffuCoder.

## Quick Start

### 1. Basic CPU Benchmark

```bash
# Run with automatic hardware detection
python jax_lm/benchmarks/hardware_benchmark.py --backend auto

# Force CPU benchmark
python jax_lm/benchmarks/hardware_benchmark.py --backend cpu --model-size small
```

### 2. PyTorch vs JAX Comparison

```bash
# Full comparison pipeline
./jax_lm/benchmarks/download_and_benchmark.sh

# Or run components separately:
python benchmarks/pytorch_cpu_benchmark.py --model-path ./models/pytorch
python jax_lm/benchmarks/hardware_benchmark.py --backend cpu
python jax_lm/benchmarks/compare_results.py \
    --pytorch-results pytorch_results.json \
    --jax-results jax_results.json
```

## Benchmark Types

### 1. Forward Pass Benchmark

Measures raw inference speed:
- Various batch sizes (1, 2, 4, 8, ...)
- Different sequence lengths (128, 256, 512, ...)
- Reports tokens/second throughput

### 2. Generation Benchmark

Tests complete generation pipeline:
- Includes diffusion sampling overhead
- Measures end-to-end generation time
- Reports time per token

### 3. Memory Benchmark

Profiles memory usage:
- Model parameter memory
- Activation memory by batch size
- Peak memory usage

### 4. Numerical Parity Test

Verifies correctness:
- Layer-by-layer comparison with PyTorch
- Checks numerical precision
- Identifies any divergence

## Hardware-Specific Benchmarks

### CPU Benchmarks

```python
# Force CPU and set thread count
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
```

Key factors:
- Number of CPU cores
- Memory bandwidth
- XLA optimization level

### GPU Benchmarks

```python
# Select specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

Optimizations:
- Mixed precision (float16/bfloat16)
- Memory pool configuration
- Kernel fusion settings

### TPU Benchmarks

```bash
# On Google Cloud TPU VM
export TPU_NAME=your-tpu-name
python jax_lm/inference_tpu.py --model-path ./models/jax
```

TPU-specific features:
- Automatic sharding
- Optimized collective operations
- Native bfloat16 support

## Interpreting Results

### Key Metrics

1. **Throughput (tokens/second)**
   - Higher is better
   - Scales with batch size
   - Primary performance metric

2. **Latency (ms/token)**
   - Lower is better
   - Important for interactive use
   - Includes all overheads

3. **Memory Efficiency**
   - MB per batch item
   - Peak vs average usage
   - Activation memory scaling

### Expected Performance Patterns

1. **Batch Size Scaling**
   - Sublinear scaling is normal
   - Larger batches = better efficiency
   - Memory limits maximum batch

2. **Sequence Length Impact**
   - Quadratic attention complexity
   - Longer sequences = slower
   - Consider chunking strategies

3. **Model Size Effects**
   - Larger models = more compute
   - Better quality vs speed tradeoff
   - Consider quantization

## Optimization Tips

### 1. JIT Compilation

Always JIT compile for best performance:
```python
@jax.jit
def generate_fn(params, input_ids):
    return model.generate(params, input_ids)
```

### 2. Batch Processing

Process multiple requests together:
```python
# Inefficient
for prompt in prompts:
    output = generate(prompt)

# Efficient
outputs = generate_batch(prompts)
```

### 3. Memory Management

Pre-allocate memory pools:
```python
# Prevent fragmentation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
```

### 4. Mixed Precision

Use appropriate precision:
```python
# For most cases
model = model.to_bf16()  # bfloat16

# For highest accuracy
model = model.to_fp32()  # float32
```

## Benchmark Scripts

### `hardware_benchmark.py`
- Comprehensive benchmark suite
- Auto-detects hardware
- Generates detailed reports

### `cpu_benchmark.py`
- CPU-specific optimizations
- Thread configuration
- Memory profiling

### `compare_results.py`
- Side-by-side comparison
- Statistical analysis
- Performance regression detection

### `test_numerical_parity.py`
- Layer-wise comparison
- Precision analysis
- Debugging tool

## Troubleshooting

### Common Issues

1. **OOM (Out of Memory)**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use model sharding

2. **Slow First Run**
   - JIT compilation overhead
   - Run warmup iterations
   - Cache compiled functions

3. **Inconsistent Results**
   - CPU throttling
   - Background processes
   - Use isolated environment

### Performance Debugging

```python
# Profile JAX operations
with jax.profiler.trace("/tmp/jax-trace"):
    output = model.generate(...)

# Analyze in TensorBoard
tensorboard --logdir=/tmp/jax-trace
```

## Reporting Benchmarks

When reporting benchmark results:

1. **Hardware Details**
   - Exact CPU/GPU/TPU model
   - Memory configuration
   - Driver versions

2. **Software Versions**
   - JAX version
   - CUDA/TPU runtime version
   - OS and kernel

3. **Benchmark Configuration**
   - Model size
   - Batch sizes tested
   - Sequence lengths
   - Number of runs

4. **Results Format**
   - Mean ± std deviation
   - Min/max values
   - Reproducibility instructions