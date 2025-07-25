# JAX DiffuCoder TPU Experiment Plan

This document provides a step-by-step plan for testing JAX DiffuCoder on TPU, with expected outcomes and troubleshooting guidance for each step.

## 🎯 Experiment Overview

**Goal**: Validate JAX DiffuCoder inference on TPU and measure performance improvements over CPU.

**Expected Outcomes**:
- Successful model loading and inference on TPU
- 2-5x performance improvement over CPU
- Numerical parity with PyTorch implementation
- Stable batch inference at various sizes

## 📋 Pre-Experiment Checklist

- [ ] JAX model weights available (`models/dream-jax/params.pkl` - 14.2GB)
- [ ] JAX config file present (`models/dream-jax/config.json`)
- [ ] Tokenizer files accessible (from PyTorch model directory)
- [ ] Google Colab Pro/Pro+ account (for TPU access) OR Cloud TPU VM

## 🧪 Experiment Steps

### Step 1: Environment Setup (5-10 min)

#### 1.1 Google Colab Setup
```python
# In new Colab notebook:
# Runtime → Change runtime type → TPU v2-8

# Cell 1: Check TPU availability
import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
# Expected: [TpuDevice(id=0..7), ...] (8 devices)

# Cell 2: Install dependencies
!pip install -q transformers sentencepiece flax
!pip install -q tqdm  # For progress bars

# Cell 3: Mount Google Drive (if model is there)
from google.colab import drive
drive.mount('/content/drive')
```

**Expected Output**:
```
JAX version: 0.4.x
Devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), ...]
```

**Troubleshooting**:
- If CPU devices shown: Runtime not set to TPU
- If import errors: Restart runtime after pip installs

#### 1.2 Cloud TPU VM Setup (Alternative)
```bash
# SSH into TPU VM
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE

# Install dependencies
pip install transformers sentencepiece flax tqdm
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Step 2: Upload Model Weights (10-15 min)

#### 2.1 For Colab
```python
# Option A: Direct upload (slow for 14GB)
from google.colab import files
uploaded = files.upload()  # Select params.pkl and config.json

# Option B: From Google Drive (recommended)
!cp "/content/drive/MyDrive/models/dream-jax/*" "./models/dream-jax/"

# Option C: Download from cloud storage
!gsutil -m cp -r gs://your-bucket/dream-jax ./models/
```

#### 2.2 Verify Files
```python
import os
model_path = "./models/dream-jax"
assert os.path.exists(f"{model_path}/params.pkl"), "params.pkl not found!"
assert os.path.exists(f"{model_path}/config.json"), "config.json not found!"

# Check file sizes
!ls -lh {model_path}/
# Expected: params.pkl ~14.2GB, config.json ~1KB
```

**Expected Output**:
```
-rw-r--r-- 1 root root 14.2G Jan 25 10:00 params.pkl
-rw-r--r-- 1 root root  517B Jan 25 10:00 config.json
```

### Step 3: Test Basic Model Loading (5 min)

```python
# Test loading without inference first
import time
import pickle
import json

# Load config
print("Loading config...")
with open("./models/dream-jax/config.json") as f:
    config = json.load(f)
print(f"Model config: {config['hidden_size']}d, {config['num_hidden_layers']} layers")

# Test pickle loading
print("\nTesting pickle load (this will take ~30s)...")
start = time.time()
with open("./models/dream-jax/params.pkl", "rb") as f:
    # Just read first few bytes to test
    test_data = f.read(1024)
print(f"Pickle file accessible, first 1KB read in {time.time()-start:.2f}s")
```

**Expected Output**:
```
Loading config...
Model config: 3584d, 28 layers

Testing pickle load...
Pickle file accessible, first 1KB read in 0.01s
```

**Troubleshooting**:
- FileNotFoundError: Check paths and file upload
- Permission denied: Check file permissions

### Step 4: Load JAX Model (2-3 min)

```python
# Upload/clone the JAX inference code
!git clone https://github.com/yourusername/ml-diffucoder.git
%cd ml-diffucoder

# Or upload the jax_lm directory
from google.colab import files
# Upload jax_lm folder as zip and extract

# Test model loading
from jax_lm.inference import DiffuCoderInference
import jax.numpy as jnp

print("Initializing model...")
start = time.time()

model = DiffuCoderInference(
    model_path="./models/dream-jax",
    tokenizer_path="./models/diffucoder-7b-complete",  # Adjust path
    dtype=jnp.bfloat16,
)

print(f"\nModel loaded in {time.time()-start:.2f}s")
```

**Expected Output**:
```
Available devices: [TpuDevice(...), ...]
Using device: tpu
Loading weights from ./models/dream-jax/params.pkl...
✅ Model loaded successfully!
   Device: tpu
   Parameters: 7.6B
   Dtype: bfloat16

Model loaded in 35.2s
```

**Troubleshooting**:
- OOM errors: TPU has 16GB HBM, model needs ~15GB
- Import errors: Ensure jax_lm is in Python path
- Tokenizer not found: Specify correct tokenizer path

### Step 5: Test Single Inference (1-2 min)

```python
# Simple generation test
prompt = "def fibonacci(n):"

print(f"Prompt: {prompt}")
print("Generating...")

output = model.generate(
    prompt,
    max_new_tokens=50,  # Start small
    temperature=0.3,
    num_steps=25,      # Fewer diffusion steps for testing
)

print(f"\nGenerated:\n{output}")
```

**Expected Output**:
```
Prompt: def fibonacci(n):
Generating...
Generating 50 tokens...
Generation completed in 4.5s
Tokens/second: 11.11

Generated:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Performance Expectations**:
- First run: 5-10s (includes JIT compilation)
- Subsequent runs: 2-5s for 50 tokens
- Tokens/second: 10-25 (will improve with larger batches)

### Step 6: Benchmark Forward Pass (2 min)

```python
# Test raw forward pass performance
results = model.benchmark_inference(
    batch_sizes=[1, 2, 4, 8],
    seq_length=128,
    num_iterations=10,
)

import json
print(json.dumps(results, indent=2))
```

**Expected Output**:
```json
{
  "batch_1": {
    "avg_time_seconds": 0.045,
    "throughput_tokens_per_second": 2844
  },
  "batch_2": {
    "avg_time_seconds": 0.048,
    "throughput_tokens_per_second": 5333
  },
  "batch_4": {
    "avg_time_seconds": 0.055,
    "throughput_tokens_per_second": 9309
  },
  "batch_8": {
    "avg_time_seconds": 0.075,
    "throughput_tokens_per_second": 13653
  }
}
```

**Key Metrics**:
- Single forward pass: ~45ms for batch_size=1
- Throughput scales sub-linearly with batch size
- Peak throughput: 10-20k tokens/second

### Step 7: Test Batch Generation (3-5 min)

```python
# Test batch generation
prompts = [
    "def quicksort(arr):",
    "class BinaryTree:",
    "def binary_search(arr, target):",
    "import numpy as np\ndef matrix_multiply(A, B):",
]

print(f"Testing batch generation with {len(prompts)} prompts...")
start = time.time()

outputs = model.generate(
    prompts,
    max_new_tokens=100,
    temperature=0.3,
    num_steps=50,
)

elapsed = time.time() - start
total_tokens = len(prompts) * 100

print(f"\nGeneration completed in {elapsed:.2f}s")
print(f"Total tokens: {total_tokens}")
print(f"Tokens/second: {total_tokens/elapsed:.2f}")

for i, (prompt, output) in enumerate(zip(prompts, outputs)):
    print(f"\n--- Example {i+1} ---")
    print(f"Prompt: {prompt}")
    print(f"Generated: {output[:100]}...")
```

**Expected Performance**:
- Batch of 4: 15-25s for 400 total tokens
- Tokens/second: 15-30
- Should see ~2-3x speedup over sequential generation

### Step 8: Stress Test with Larger Batches (5 min)

```python
# Test larger batches to find optimal size
import numpy as np

batch_sizes = [1, 2, 4, 8, 16, 32]
results = []

for bs in batch_sizes:
    try:
        print(f"\nTesting batch size {bs}...")
        
        # Create dummy prompts
        prompts = ["def test():" for _ in range(bs)]
        
        # Time generation
        start = time.time()
        _ = model.generate(
            prompts,
            max_new_tokens=50,
            temperature=0.3,
            num_steps=25,
        )
        elapsed = time.time() - start
        
        tokens_per_sec = (bs * 50) / elapsed
        results.append({
            "batch_size": bs,
            "time": elapsed,
            "tokens_per_sec": tokens_per_sec,
        })
        
        print(f"  Time: {elapsed:.2f}s, Tokens/sec: {tokens_per_sec:.2f}")
        
    except Exception as e:
        print(f"  Failed with error: {e}")
        break

# Plot results
import matplotlib.pyplot as plt

batch_sizes = [r["batch_size"] for r in results]
tokens_per_sec = [r["tokens_per_sec"] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, tokens_per_sec, 'bo-', linewidth=2, markersize=8)
plt.xlabel("Batch Size")
plt.ylabel("Tokens/Second")
plt.title("TPU Throughput vs Batch Size")
plt.grid(True)
plt.show()
```

**Expected Results**:
- Optimal batch size: 8-16 for TPU v2-8
- Peak throughput: 50-100 tokens/second
- Performance plateaus or decreases beyond optimal size

### Step 9: Compare with CPU Performance (Optional, 10 min)

```python
# Switch to CPU for comparison
import os
os.environ["JAX_PLATFORMS"] = "cpu"

# Restart runtime and reload
import jax
print(f"Now using: {jax.devices()}")

# Re-run same test on CPU
# ... (same code as Step 5)
```

**Expected CPU Performance**:
- 50 tokens: 2-5 minutes
- Tokens/second: 0.2-0.5
- TPU speedup: 50-200x

### Step 10: Numerical Validation (5 min)

```python
# If PyTorch model is available, run parity test
!python jax_lm/tests/test_layer_parity.py \
    --pytorch-model ./models/diffucoder-7b-complete \
    --jax-model ./models/dream-jax \
    --tolerance 1e-4
```

**Expected Results**:
- Embedding layer: Pass (exact match)
- RMSNorm layers: Pass (within 1e-4)
- Attention projections: Pass (after transposition)
- Overall pass rate: >95%

## 📊 Results Summary Template

After completing all steps, fill in this summary:

```markdown
## Experiment Results - [DATE]

### Environment
- Platform: Google Colab TPU / Cloud TPU VM
- TPU Type: v2-8 / v3-8 / v4-8
- JAX Version: X.X.X
- Model: DiffuCoder 7.6B

### Performance Metrics
- Model Load Time: XXs
- Single Token Generation: XXms
- Batch 1 Throughput: XX tokens/sec
- Batch 8 Throughput: XX tokens/sec
- Optimal Batch Size: XX
- Peak Throughput: XX tokens/sec

### Comparison
- CPU Throughput: XX tokens/sec
- TPU Speedup: XXx

### Issues Encountered
- [ ] None
- [ ] OOM at batch size XX
- [ ] JIT compilation timeout
- [ ] Other: ___________

### Next Steps
- [ ] Test larger sequences
- [ ] Profile memory usage
- [ ] Optimize generation parameters
```

## 🚨 Common Issues & Solutions

### 1. Out of Memory (OOM)
```
XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error
```
**Solution**: Reduce batch size or use gradient checkpointing

### 2. Slow First Generation
**Cause**: JIT compilation on first run
**Solution**: Normal - subsequent runs will be fast

### 3. Import Errors
```
ModuleNotFoundError: No module named 'jax_lm'
```
**Solution**: Ensure jax_lm is in Python path or install as package

### 4. Tokenizer Issues
```
OSError: Can't load tokenizer
```
**Solution**: Upload tokenizer files or specify correct path

## 🎯 Success Criteria

The experiment is successful if:
1. ✅ Model loads on TPU without errors
2. ✅ Single inference completes in <10s
3. ✅ Batch inference shows speedup
4. ✅ TPU is >10x faster than CPU
5. ✅ Generation quality is coherent

## 📈 Performance Optimization Tips

1. **Batch Size**: Start with 8, adjust based on memory
2. **Sequence Length**: Shorter sequences = higher throughput
3. **Diffusion Steps**: Reduce for faster but lower quality
4. **Temperature**: Lower = more deterministic, faster
5. **Dtype**: bfloat16 is optimal for TPU

## 🔗 Resources

- [JAX TPU Setup](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)
- [Colab TPU Guide](https://colab.research.google.com/notebooks/tpu.ipynb)
- [Cloud TPU Docs](https://cloud.google.com/tpu/docs)
- [JAX Profiling](https://jax.readthedocs.io/en/latest/profiling.html)