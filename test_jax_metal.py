#!/usr/bin/env python3
"""
Test JAX with Metal backend, loading weights on CPU first.
"""

import os
import sys
import time
import json
from pathlib import Path

# Temporarily disable Metal to load weights
os.environ["JAX_PLATFORMS"] = "cpu"

# Add jax_lm to path
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

import jax
import jax.numpy as jnp
from jax import random
from transformers import AutoTokenizer

print("Loading weights on CPU first...")
print(f"JAX devices (CPU): {jax.devices()}")

from jax_lm.models.dream import DreamConfig, DreamForCausalLM

# Load config
print("\n📥 Loading model config...")
with open("./models/dream-jax/config.json") as f:
    config_dict = json.load(f)

config = DreamConfig(**{k: v for k, v in config_dict.items() if k in [
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
    "rope_theta", "rms_norm_eps", "attention_dropout"
]})

print(f"Model config: {config.hidden_size}d hidden, {config.num_hidden_layers} layers")

# Load weights on CPU
print("📦 Loading weights on CPU...")
import pickle
start = time.time()
with open("./models/dream-jax/params.pkl", "rb") as f:
    params_cpu = pickle.load(f)
print(f"✅ Weights loaded in {time.time() - start:.2f}s")

# Count parameters
num_params = sum(x.size for x in jax.tree_util.tree_leaves(params_cpu))
print(f"📊 Model parameters: {num_params:,} ({num_params/1e9:.1f}B)")

# Now re-initialize JAX with Metal
print("\n🔄 Switching to Metal backend...")
# Force JAX to reinitialize
if hasattr(jax._src.xla_bridge, "_backends"):
    jax._src.xla_bridge._backends = {}

# Re-import to use Metal
os.environ["JAX_PLATFORMS"] = ""  # Reset to default
import importlib
importlib.reload(jax)
importlib.reload(jax._src.xla_bridge)

# Check Metal is available
print(f"JAX devices (Metal): {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Transfer weights to Metal
print("\n🚀 Transferring weights to Metal...")
start = time.time()
params = jax.tree_map(lambda x: jax.device_put(x), params_cpu)
print(f"✅ Weights transferred in {time.time() - start:.2f}s")

# Create model on Metal
print("\n🔨 Creating model on Metal...")
model = DreamForCausalLM(config=config, dtype=jnp.float32)  # Use float32 for Metal

# Load tokenizer
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "./models/diffucoder-7b-complete",
    trust_remote_code=True,
    local_files_only=True,
)

# Quick test
prompt = "def hello():"
print(f"\n🧪 Testing with prompt: '{prompt}'")

# Tokenize
tokens = tokenizer.encode(prompt)
input_ids = jnp.array([tokens])
print(f"📝 Input tokens: {tokens} (length: {len(tokens)})")

# JIT compile forward pass
print("\n⚡ JIT compiling for Metal...")
@jax.jit
def forward_fn(params, input_ids):
    return model.apply(params, input_ids, deterministic=True)

# Warmup
print("🔥 Warmup...")
start = time.time()
try:
    _ = forward_fn(params, input_ids)
    print(f"✅ First forward pass: {time.time() - start:.2f}s")
    
    # Time a few forward passes
    print("\n⏱️  Timing forward passes...")
    times = []
    for i in range(5):
        start = time.time()
        outputs = forward_fn(params, input_ids)
        outputs["logits"].block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Pass {i+1}: {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Average forward pass time: {avg_time:.3f}s")
    print(f"💡 Estimated tokens/second: {1/avg_time:.2f}")
    
except Exception as e:
    print(f"❌ Error during inference: {e}")
    print("\nMetal backend may not support all operations needed for this model.")
    print("This is expected as JAX Metal support is still experimental.")

print("\n🏁 Test complete!")