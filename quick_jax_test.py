#!/usr/bin/env python3
"""
Quick JAX test with minimal token generation to verify functionality.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add jax_lm to path
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

import jax
import jax.numpy as jnp
from jax import random
from transformers import AutoTokenizer

from jax_lm.models.dream import DreamConfig, DreamForCausalLM

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Load config
print("\n📥 Loading model config...")
with open("./models/dream-jax/config.json") as f:
    config_dict = json.load(f)

config = DreamConfig(**{k: v for k, v in config_dict.items() if k in [
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
    "rope_theta", "rms_norm_eps", "attention_dropout"
]})

# Create model
print("🔨 Creating model...")
model = DreamForCausalLM(config=config, dtype=jnp.bfloat16)

# Load weights
print("📦 Loading weights...")
import pickle
start = time.time()
with open("./models/dream-jax/params.pkl", "rb") as f:
    params = pickle.load(f)
print(f"✅ Weights loaded in {time.time() - start:.2f}s")

# Count parameters
num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"📊 Model parameters: {num_params:,} ({num_params/1e9:.1f}B)")

# Load tokenizer
print("\n🔤 Loading tokenizer...")
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
print("\n⚡ JIT compiling...")
@jax.jit
def forward_fn(params, input_ids):
    return model.apply(params, input_ids, deterministic=True)

# Warmup
print("🔥 Warmup...")
start = time.time()
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

# Generate just 5 tokens
print("\n🎯 Generating 5 tokens...")
rng = random.PRNGKey(42)
current_ids = input_ids

@jax.jit
def generate_step(params, input_ids, rng_key):
    outputs = model.apply(params, input_ids, deterministic=True)
    logits = outputs["logits"][:, -1, :]
    next_token = jnp.argmax(logits, axis=-1)
    return jnp.expand_dims(next_token, axis=0)

generated_tokens = []
for i in range(5):
    print(f"  Generating token {i+1}/5...", end="", flush=True)
    start = time.time()
    rng, step_rng = random.split(rng)
    next_token = generate_step(params, current_ids, step_rng)
    current_ids = jnp.concatenate([current_ids, next_token], axis=-1)
    current_ids.block_until_ready()
    elapsed = time.time() - start
    generated_tokens.append(int(next_token[0, 0]))
    print(f" {elapsed:.2f}s")

# Decode
full_tokens = tokens + generated_tokens
generated_text = tokenizer.decode(full_tokens)
print(f"\n✅ Generated text: '{generated_text}'")
print(f"📝 Generated tokens: {generated_tokens}")

print("\n🏁 Test complete!")