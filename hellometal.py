#!/usr/bin/env python3
"""
Simplest possible JAX Metal test - just basic array operations.
"""

import os
# Ensure we're using Metal
os.environ["JAX_PLATFORMS"] = "METAL"

import jax
import jax.numpy as jnp

print("🍎 Hello JAX Metal!")
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")

# Test 1: Simple array creation
print("\n📊 Test 1: Create array")
try:
    # Create array directly on device
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"✅ Created array: {x}")
    print(f"   Shape: {x.shape}, dtype: {x.dtype}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Simple arithmetic
print("\n📊 Test 2: Simple arithmetic")
try:
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    c = a + b
    print(f"✅ a + b = {c}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 3: Small matrix multiply
print("\n📊 Test 3: Small matrix multiply")
try:
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    C = jnp.dot(A, B)
    print(f"✅ Matrix multiply result:\n{C}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: JIT compilation
print("\n📊 Test 4: JIT compilation")
try:
    @jax.jit
    def add_one(x):
        return x + 1
    
    x = jnp.array([1.0, 2.0, 3.0])
    y = add_one(x)
    print(f"✅ JIT compiled function result: {y}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 5: Device info
print("\n📊 Test 5: Device info")
try:
    device = jax.devices()[0]
    print(f"✅ Device: {device}")
    print(f"   Platform: {device.platform}")
    print(f"   Device kind: {device.device_kind}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n🏁 Tests complete!")