"""
Test notebook for jax-diffucoder
Copy these cells to a Google Colab notebook
"""

# Cell 1: Install directly from GitHub (no auth needed for public repo)
!pip install git+https://github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm

# Cell 2: Verify installation
import jax
import jax_lm
print(f"✓ jax-diffucoder version: {jax_lm.__version__}")
print(f"✓ JAX version: {jax.__version__}")
print(f"✓ JAX devices: {jax.devices()}")

# Cell 3: Test imports
from jax_lm import DiffuCoderConfig, load_model, generate
print("✓ All imports successful")

# Cell 4: Test configuration
config = DiffuCoderConfig()
print(f"Model config:")
print(f"  - hidden_size: {config.hidden_size}")
print(f"  - num_layers: {config.num_hidden_layers}")
print(f"  - num_heads: {config.num_attention_heads}")

# Cell 5: Test model loading (will fail until HF upload is done)
try:
    model, params, tokenizer = load_model("atsentia/DiffuCoder-7B-JAX")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed (expected until HF upload): {e}")

# Cell 6: Check if running on TPU
if len(jax.devices("tpu")) > 0:
    print(f"🚀 Running on TPU with {len(jax.devices('tpu'))} cores")
else:
    print(f"💻 Running on {jax.devices()[0].device_kind}")