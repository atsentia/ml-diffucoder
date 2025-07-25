"""Flax implementation of DiffuCoder model."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import compact
from flax.core import freeze, unfreeze
import einops

from jax_lm.models.components import (
    TransformerBlock,
    RMSNorm,
    PositionalEncoding,
    create_causal_mask,
)


@dataclass
class DiffuCoderConfig:
    """Configuration for DiffuCoder model."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    
    # Diffusion-specific parameters
    diffusion_steps: int = 256
    mask_token_id: int = 32001  # Special token for masking
    pad_token_id: int = 32002
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class DiffuCoderEmbedding(nn.Module):
    """Token and position embeddings for DiffuCoder."""
    config: DiffuCoderConfig
    dtype: Any = jnp.float32
    
    @compact
    def __call__(self, input_ids, position_ids=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        embed = nn.Embed(
            num_embeddings=self.config.vocab_size + 2,  # +2 for mask and pad tokens
            features=self.config.hidden_size,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=self.config.initializer_range),
        )
        token_embeds = embed(input_ids)
        
        # Position embeddings using RoPE are applied in attention layers
        return token_embeds


class DiffuCoderMLP(nn.Module):
    """MLP layer for DiffuCoder."""
    config: DiffuCoderConfig
    dtype: Any = jnp.float32
    
    @compact
    def __call__(self, x):
        gate_proj = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=False,
        )
        up_proj = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=False,
        )
        down_proj = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=False,
        )
        
        if self.config.hidden_act == "silu":
            act_fn = nn.silu
        elif self.config.hidden_act == "gelu":
            act_fn = nn.gelu
        else:
            raise ValueError(f"Unsupported activation: {self.config.hidden_act}")
        
        return down_proj(act_fn(gate_proj(x)) * up_proj(x))


class DiffuCoderAttention(nn.Module):
    """Multi-head attention with RoPE for DiffuCoder."""
    config: DiffuCoderConfig
    dtype: Any = jnp.float32
    
    def setup(self):
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        
        self.q_proj = nn.Dense(
            self.config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.attention_bias,
        )
        self.k_proj = nn.Dense(
            self.config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.attention_bias,
        )
        self.v_proj = nn.Dense(
            self.config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.attention_bias,
        )
        self.o_proj = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=self.config.attention_bias,
        )
        
        if self.config.attention_dropout > 0:
            self.dropout = nn.Dropout(rate=self.config.attention_dropout)
    
    def _apply_rotary_pos_emb(self, x, position_ids):
        """Apply rotary position embeddings."""
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        # Create rotation angles
        dim_half = head_dim // 2
        inv_freq = 1.0 / (self.config.rope_theta ** (jnp.arange(0, dim_half, 2) / dim_half))
        
        # Compute sin and cos
        position_ids = position_ids[:, :, None]  # [batch, seq_len, 1]
        freqs = position_ids * inv_freq[None, None, :]  # [batch, seq_len, dim_half//2]
        
        # Duplicate for sin and cos
        freqs = jnp.repeat(freqs, 2, axis=-1)  # [batch, seq_len, dim_half]
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        # Apply rotation
        x_rot = x[..., :dim_half]
        x_pass = x[..., dim_half:]
        
        # Reshape for rotation
        x_rot = einops.rearrange(x_rot, "b s h (d r) -> b s h d r", r=2)
        x_rot_new = jnp.stack([
            x_rot[..., 0] * cos[:, :, None, :] - x_rot[..., 1] * sin[:, :, None, :],
            x_rot[..., 0] * sin[:, :, None, :] + x_rot[..., 1] * cos[:, :, None, :],
        ], axis=-1)
        x_rot_new = einops.rearrange(x_rot_new, "b s h d r -> b s h (d r)")
        
        return jnp.concatenate([x_rot_new, x_pass], axis=-1)
    
    @compact
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, deterministic=True):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        query_states = query_states.reshape(batch_size, seq_len, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))
        
        query_states = self._apply_rotary_pos_emb(query_states, position_ids)
        key_states = self._apply_rotary_pos_emb(key_states, position_ids)
        
        # Repeat k/v heads if necessary
        if self.num_key_value_groups > 1:
            key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
            value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)
        
        # Compute attention scores
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bshd,bthd->bhst", query_states, key_states) * scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = nn.softmax(scores, axis=-1)
        
        # Apply dropout
        if self.config.attention_dropout > 0 and not deterministic:
            attn_weights = self.dropout(attn_weights, deterministic=deterministic)
        
        # Compute attention output
        attn_output = jnp.einsum("bhst,bthd->bshd", attn_weights, value_states)
        attn_output = attn_output.reshape(batch_size, seq_len, self.config.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class DiffuCoderBlock(nn.Module):
    """Transformer block for DiffuCoder."""
    config: DiffuCoderConfig
    dtype: Any = jnp.float32
    
    @compact
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, deterministic=True):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = RMSNorm(self.config.rms_norm_eps, dtype=self.dtype)(hidden_states)
        hidden_states = DiffuCoderAttention(self.config, dtype=self.dtype)(
            hidden_states, attention_mask, position_ids, deterministic
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = RMSNorm(self.config.rms_norm_eps, dtype=self.dtype)(hidden_states)
        hidden_states = DiffuCoderMLP(self.config, dtype=self.dtype)(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DiffuCoder(nn.Module):
    """DiffuCoder model for masked diffusion language modeling."""
    config: DiffuCoderConfig
    dtype: Any = jnp.float32
    
    @compact
    def __call__(
        self, 
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        return_dict=True,
    ):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = DiffuCoderEmbedding(self.config, dtype=self.dtype)(input_ids, position_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        # Convert attention mask to 4D for compatibility
        attention_mask = create_causal_mask(attention_mask, self.dtype)
        
        # Apply transformer blocks
        for i in range(self.config.num_hidden_layers):
            hidden_states = DiffuCoderBlock(self.config, dtype=self.dtype, name=f"layer_{i}")(
                hidden_states, attention_mask, position_ids, deterministic
            )
        
        # Final layer norm
        hidden_states = RMSNorm(self.config.rms_norm_eps, dtype=self.dtype, name="norm")(hidden_states)
        
        # Language modeling head
        if self.config.tie_word_embeddings:
            # Reuse embedding weights
            embed_layer = self.variables['params']['DiffuCoderEmbedding_0']['Embed_0']['embedding']
            lm_logits = jnp.dot(hidden_states, embed_layer.T)
        else:
            lm_head = nn.Dense(
                self.config.vocab_size + 2,
                dtype=self.dtype,
                kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
                use_bias=False,
                name="lm_head",
            )
            lm_logits = lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": lm_logits,
                "hidden_states": hidden_states,
            }
        return lm_logits