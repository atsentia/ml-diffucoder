#!/usr/bin/env python3
"""
Simple test of JAX Metal performance with unified memory.
"""

import jax
import jax.numpy as jnp
from jax import random
import time

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Test basic operations
print("\n📊 Testing basic operations on Metal...")

# Create some test data
key = random.PRNGKey(42)
size = (1024, 1024)

# Matrix multiplication test
print("\n🔢 Matrix multiplication test:")
A = random.normal(key, size)
B = random.normal(key, size)

# JIT compile
@jax.jit
def matmul(A, B):
    return jnp.dot(A, B)

# Warmup
_ = matmul(A, B).block_until_ready()

# Time it
times = []
for i in range(10):
    start = time.time()
    C = matmul(A, B).block_until_ready()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed*1000:.2f}ms")

avg_time = sum(times) / len(times)
print(f"\nAverage time: {avg_time*1000:.2f}ms")
print(f"GFLOPS: {2 * size[0]**3 / (avg_time * 1e9):.2f}")

# Test with bfloat16
print("\n🔢 Testing with bfloat16:")
A_bf16 = A.astype(jnp.bfloat16)
B_bf16 = B.astype(jnp.bfloat16)

@jax.jit
def matmul_bf16(A, B):
    return jnp.dot(A, B)

# Warmup
_ = matmul_bf16(A_bf16, B_bf16).block_until_ready()

# Time it
times_bf16 = []
for i in range(10):
    start = time.time()
    C = matmul_bf16(A_bf16, B_bf16).block_until_ready()
    elapsed = time.time() - start
    times_bf16.append(elapsed)

avg_time_bf16 = sum(times_bf16) / len(times_bf16)
print(f"Average time (bfloat16): {avg_time_bf16*1000:.2f}ms")
print(f"Speedup vs float32: {avg_time/avg_time_bf16:.2f}x")

# Check if we can handle the model size
model_params = 7.6e9  # 7.6B parameters
bytes_per_param = 2  # bfloat16
model_size_gb = (model_params * bytes_per_param) / 1e9

print(f"\n💾 Model memory requirements:")
print(f"   Parameters: {model_params/1e9:.1f}B")
print(f"   Size (bfloat16): {model_size_gb:.1f} GB")
print(f"   Available Metal memory: ~10.67 GB")
print(f"   Fits in Metal memory: {'Yes' if model_size_gb < 10.67 else 'No (will use shared memory)'}")

print("\n✅ Metal backend is working!")