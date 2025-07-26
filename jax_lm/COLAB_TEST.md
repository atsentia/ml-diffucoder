# Testing jax-diffucoder on Google Colab

## Quick Start

Run these cells in a Google Colab notebook:

### 1. Install the package

```python
!pip install jax-diffucoder
```

### 2. Basic import test

```python
import jax
import jax.numpy as jnp
from jax_lm import load_model, generate

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
```

### 3. Load model (once uploaded to HuggingFace)

```python
# This will work once the model is uploaded to HuggingFace
model, params, tokenizer = load_model("atsentia/DiffuCoder-7B-JAX")

# Test generation
prompt = "def fibonacci(n):"
output = generate(model, params, prompt, tokenizer, max_new_tokens=100)
print(output)
```

### 4. TPU-specific setup (if using TPU runtime)

```python
# For TPU runtime in Colab
import jax

# Check TPU availability
if len(jax.devices("tpu")) > 0:
    print(f"TPU detected: {jax.devices('tpu')}")
    print(f"Number of TPU cores: {len(jax.devices('tpu'))}")
else:
    print("No TPU detected, using CPU/GPU")
```

## Note

The model weights need to be uploaded to HuggingFace first. Currently they are available locally at:
- `/Users/amund/ml-diffucoder/models/DiffuCoder-7B-JAX/`
- Total size: 11.85 GB

Once uploaded, users will be able to load the model directly from HuggingFace Hub.