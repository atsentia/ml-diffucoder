# DiffuCoder Architecture

This document provides a detailed overview of the JAX/Flax DiffuCoder architecture and its implementation details.

## Overview

DiffuCoder is a masked diffusion language model designed specifically for code generation. Unlike traditional autoregressive models, DiffuCoder generates code through an iterative refinement process using masked token prediction.

## Core Components

### 1. Model Architecture

The DiffuCoder architecture consists of:

```
Input Tokens → Embeddings → Transformer Blocks → Output Logits
                    ↓
              [RoPE Positional Encoding]
```

#### Key Components:

- **Embedding Layer**: Maps token IDs to dense vectors
  - Vocabulary size: 32,000 + 2 special tokens (mask, pad)
  - Hidden dimension: 4096 (7B model)

- **Transformer Blocks**: 32 layers (7B model)
  - Multi-head attention with RoPE
  - RMSNorm for layer normalization
  - SwiGLU activation in FFN layers

- **Output Head**: Projects hidden states back to vocabulary

### 2. Rotary Position Embeddings (RoPE)

RoPE provides relative position information without explicit position embeddings:

```python
def apply_rope(x, position_ids):
    # Compute rotation frequencies
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim//2, 2) / dim))
    
    # Apply rotation to query and key vectors
    cos = np.cos(position_ids * inv_freq)
    sin = np.sin(position_ids * inv_freq)
    
    # Rotate embeddings
    x_rot = rotate_half(x)
    x = x * cos + x_rot * sin
    return x
```

### 3. Attention Mechanism

Multi-head attention with grouped query attention (GQA):

```python
class DiffuCoderAttention(nn.Module):
    def __call__(self, hidden_states, attention_mask=None):
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Apply RoPE
        query = apply_rope(query, position_ids)
        key = apply_rope(key, position_ids)
        
        # Compute attention
        scores = einsum("bshd,bthd->bhst", query, key) / sqrt(head_dim)
        attn_weights = softmax(scores + attention_mask)
        
        # Apply attention
        output = einsum("bhst,bthd->bshd", attn_weights, value)
        return self.o_proj(output)
```

### 4. Feed-Forward Network

SwiGLU activation function for improved performance:

```python
class DiffuCoderMLP(nn.Module):
    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU activation
        hidden = silu(gate) * up
        
        return self.down_proj(hidden)
```

## Diffusion Process

### 1. Masked Token Prediction

DiffuCoder generates text by iteratively unmasking tokens:

1. Start with all output positions masked
2. Predict tokens for all positions simultaneously
3. Unmask the most confident predictions
4. Repeat until all tokens are generated

### 2. Entropy-Based Unmasking

The unmasking strategy uses entropy to identify confident predictions:

```python
def compute_entropy_mask(logits, mask):
    # Compute token probabilities
    probs = softmax(logits)
    
    # Calculate entropy for masked positions
    entropy = -sum(probs * log(probs))
    entropy = where(mask == 1, entropy, -inf)
    
    # Select positions with lowest entropy (highest confidence)
    return entropy
```

### 3. Generation Algorithm

```python
def diffusion_generate(model, params, input_ids, num_steps):
    # Initialize with masked tokens
    sequences = concatenate([input_ids, mask_tokens])
    mask = create_mask(sequences)
    
    for step in range(num_steps):
        # Forward pass
        logits = model(sequences)
        
        # Sample tokens
        sampled_tokens = sample_from_logits(logits)
        
        # Update masked positions
        sequences = where(mask, sampled_tokens, sequences)
        
        # Determine positions to unmask
        entropy = compute_entropy_mask(logits, mask)
        positions_to_unmask = argmin(entropy, k=tokens_per_step)
        
        # Update mask
        mask[positions_to_unmask] = 0
        
        if sum(mask) == 0:
            break
    
    return sequences
```

## JAX-Specific Optimizations

### 1. JIT Compilation

All core functions are JIT-compiled for performance:

```python
@jax.jit
def forward_pass(params, input_ids):
    return model.apply(params, input_ids)
```

### 2. Efficient Attention

JAX implementation uses optimized einsum operations:

```python
# Efficient batched matrix multiplication
scores = jnp.einsum("bshd,bthd->bhst", query, key)
```

### 3. Memory-Efficient Gradient Computation

Using JAX's gradient checkpointing:

```python
@partial(jax.checkpoint, policy=checkpoint_policy)
def transformer_block(x, params):
    # Recompute activations during backward pass
    return block(x, params)
```

## Model Variants

### 1. Base Model
- 7B parameters
- Trained on code corpus
- No instruction tuning

### 2. Instruct Model
- Fine-tuned for instruction following
- Supports chat-style interactions
- Code-specific optimizations

### 3. cpGRPO Model
- Further optimized with Coupled-GRPO
- Better code quality
- Improved syntax awareness

## Configuration

Key configuration parameters:

```python
@dataclass
class DiffuCoderConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For GQA
    intermediate_size: int = 11008
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    
    # Diffusion specific
    diffusion_steps: int = 256
    mask_token_id: int = 32001
    pad_token_id: int = 32002
```

## Performance Considerations

### 1. Batch Processing
- Efficient batching of multiple sequences
- Dynamic padding to minimize wasted computation

### 2. Mixed Precision
- Support for bfloat16/float16
- Automatic mixed precision training

### 3. Distributed Training
- Data parallelism across devices
- Model parallelism for very large models

## Future Improvements

1. **Adaptive Diffusion Steps**: Dynamically adjust steps based on complexity
2. **Syntax-Aware Masking**: Mask complete syntactic units
3. **Multi-Modal Support**: Extend to handle code + documentation
4. **Retrieval Augmentation**: Integrate with code search