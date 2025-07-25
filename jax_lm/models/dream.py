"""Pure JAX/Flax implementation of the Dream model (DiffuCoder architecture)."""

import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
import optax

from dataclasses import dataclass


@dataclass
class DreamConfig:
    """Configuration for Dream model."""
    vocab_size: int = 152064 
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Any] = None
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    mask_token_id: int = 151666
    pad_token_id: int = 151643
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    dtype: jnp.dtype = jnp.bfloat16


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create causal attention mask."""
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask[None, None, :, :]  # Add batch and head dimensions


def apply_attention_mask(attention_mask: Optional[jnp.ndarray], causal_mask: jnp.ndarray) -> jnp.ndarray:
    """Combine causal mask with optional attention mask."""
    if attention_mask is not None:
        # Expand attention mask to match causal mask shape
        expanded_mask = attention_mask[:, None, None, :]
        combined_mask = causal_mask * expanded_mask
        return combined_mask
    return causal_mask


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings to query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].astype(q.dtype)
        sin = sin[position_ids].astype(q.dtype)
    else:
        cos = cos.astype(q.dtype)
        sin = sin.astype(q.dtype)
    
    # Ensure cos and sin have the correct dimensions for broadcasting
    # q and k have shape [batch, seq_len, num_heads, head_dim]
    # cos and sin should have shape [batch, seq_len, head_dim] or [seq_len, head_dim]
    
    # If cos/sin are 2D [seq_len, head_dim], add batch dimension
    if cos.ndim == 2:
        cos = cos[None, :, :]  # [1, seq_len, head_dim]
        sin = sin[None, :, :]  # [1, seq_len, head_dim]
    
    # Add head dimension for broadcasting: [batch, seq_len, 1, head_dim]
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
        
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def scaled_dot_product_attention(query, key, value, mask=None, dropout_rate=0.0, deterministic=True, rng=None):
    """Pure JAX implementation of scaled dot-product attention."""
    # Compute attention scores
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale
    
    # Apply mask
    if mask is not None:
        # Convert boolean mask to attention mask (0 for masked, large negative for unmasked)
        mask = jnp.where(mask, 0.0, -1e9)
        scores = scores + mask
    
    # Softmax
    attention_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply dropout if training
    if not deterministic and dropout_rate > 0.0:
        if rng is None:
            raise ValueError("RNG key required for dropout during training")
        attention_weights = nn.Dropout(rate=dropout_rate)(attention_weights, deterministic=False, rng=rng)
    
    # Apply attention to values
    output = jnp.einsum('bhqk,bhkd->bhqd', attention_weights, value)
    
    return output


class DreamRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, hidden_states):
        variance = jnp.mean(jnp.square(hidden_states.astype(jnp.float32)), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        
        weight = self.param("weight", nn.initializers.ones, (hidden_states.shape[-1],))
        return weight * hidden_states.astype(self.dtype)


class DreamRotaryEmbedding(nn.Module):
    """Rotary position embeddings."""
    
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000
    
    def setup(self):
        # Compute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        self.inv_freq = inv_freq
    
    def __call__(self, seq_len: int):
        # Create position indices
        t = jnp.arange(seq_len, dtype=jnp.float32)
        
        # Compute frequencies
        freqs = jnp.outer(t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        
        return cos, sin


class DreamMLP(nn.Module):
    """MLP module for Dream model."""
    
    config: DreamConfig
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, hidden_states):
        gate_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )(hidden_states)
        
        up_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )(hidden_states)
        
        down_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )(nn.silu(gate_proj) * up_proj)
        
        return down_proj


class DreamAttention(nn.Module):
    """Multi-headed attention module for Dream model."""
    
    config: DreamConfig
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )
        self.o_proj = nn.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )
        
        # Rotary embeddings
        self.rotary_emb = DreamRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
    
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, deterministic=True):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(seq_len)
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]
        
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # Repeat key and value states for grouped-query attention
        if self.num_key_value_groups > 1:
            key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
            value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)
        
        # Transpose for attention computation
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))  # (batch, heads, seq_len, head_dim)
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len)
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, self.num_heads, seq_len, seq_len))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            causal_mask = apply_attention_mask(attention_mask, causal_mask)
        
        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            mask=causal_mask,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
        )
        
        # Reshape and project output
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # (batch, seq_len, heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class DreamDecoderLayer(nn.Module):
    """Transformer decoder layer for Dream model."""
    
    config: DreamConfig
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, deterministic=True):
        residual = hidden_states
        
        # Self-attention
        hidden_states = DreamRMSNorm(eps=self.config.rms_norm_eps, dtype=self.dtype)(hidden_states)
        hidden_states = DreamAttention(config=self.config, dtype=self.dtype)(
            hidden_states, attention_mask, position_ids, deterministic
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = DreamRMSNorm(eps=self.config.rms_norm_eps, dtype=self.dtype)(hidden_states)
        hidden_states = DreamMLP(config=self.config, dtype=self.dtype)(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DreamModel(nn.Module):
    """Dream transformer model."""
    
    config: DreamConfig
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, input_ids, attention_mask=None, position_ids=None, deterministic=True):
        # Embeddings
        embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        hidden_states = embed_tokens(input_ids)
        
        # Transformer layers
        for i in range(self.config.num_hidden_layers):
            hidden_states = DreamDecoderLayer(
                config=self.config, 
                dtype=self.dtype,
                name=f"layers_{i}"
            )(hidden_states, attention_mask, position_ids, deterministic)
        
        # Final norm
        hidden_states = DreamRMSNorm(eps=self.config.rms_norm_eps, dtype=self.dtype)(hidden_states)
        
        return hidden_states


class DreamForCausalLM(nn.Module):
    """Dream model for causal language modeling."""
    
    config: DreamConfig
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, input_ids, attention_mask=None, position_ids=None, deterministic=True):
        # Base model
        hidden_states = DreamModel(config=self.config, dtype=self.dtype)(
            input_ids, attention_mask, position_ids, deterministic
        )
        
        # Language modeling head
        if self.config.tie_word_embeddings:
            # Use shared embeddings for output projection
            embed_tokens = self.get_variable("params", "DreamModel_0", "embed_tokens", "embedding")
            logits = jnp.dot(hidden_states, embed_tokens.T)
        else:
            lm_head = nn.Dense(
                self.config.vocab_size,
                use_bias=False,
                dtype=self.dtype,
                kernel_init=nn.initializers.normal(self.config.initializer_range),
            )
            logits = lm_head(hidden_states)
        
        return {"logits": logits, "hidden_states": hidden_states}


def create_dream_model(config: DreamConfig):
    """Create a Dream model with the given configuration."""
    return DreamForCausalLM(config=config, dtype=config.dtype)


def init_dream_model(config: DreamConfig, rng_key: random.PRNGKey, input_shape: Tuple[int, int]):
    """Initialize Dream model parameters."""
    model = create_dream_model(config)
    
    # Create dummy inputs for initialization
    dummy_input_ids = jnp.ones(input_shape, dtype=jnp.int32)
    
    # Initialize parameters
    params = model.init(rng_key, dummy_input_ids)
    
    return model, params