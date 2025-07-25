"""Common components for transformer models."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    eps: float = 1e-6
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, x.shape[-1], self.dtype)
        
        var = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        return x * weight


class TransformerBlock(nn.Module):
    """Generic transformer block (placeholder for now)."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, mask=None):
        # This is a placeholder - actual implementation would go here
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    max_len: int = 5000
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, d_model = x.shape
        
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        
        pe = jnp.zeros((seq_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return x + pe[None, :seq_len, :].astype(self.dtype)


def create_causal_mask(attention_mask, dtype):
    """Create a causal attention mask from a 2D mask."""
    batch_size, seq_len = attention_mask.shape
    
    # Create causal mask
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))
    causal_mask = causal_mask[None, None, :, :]  # Add batch and head dimensions
    
    # Combine with attention mask
    attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
    attention_mask = attention_mask * attention_mask.transpose(0, 1, 3, 2)  # [batch, 1, seq_len, seq_len]
    
    # Combine causal and attention masks
    mask = causal_mask * attention_mask
    
    # Convert to additive mask (0 -> -inf, 1 -> 0)
    mask = jnp.where(mask == 0, -1e9, 0.0).astype(dtype)
    
    return mask