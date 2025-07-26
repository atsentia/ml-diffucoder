"""Diffusion generation utilities for JAX/Flax DiffuCoder."""

from typing import Optional, Dict, Any, Tuple, Callable
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig


def compute_entropy_mask(
    logits: jnp.ndarray,
    mask: jnp.ndarray,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> jnp.ndarray:
    """Compute entropy-based mask for next tokens to unmask.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        mask: Current mask [batch_size, seq_len] (1 = masked, 0 = unmasked)
        temperature: Temperature for entropy computation
        top_p: Top-p threshold for nucleus sampling
        
    Returns:
        Binary mask indicating which positions to unmask next
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Compute probabilities
    probs = jax.nn.softmax(scaled_logits, axis=-1)
    
    # Apply top-p filtering
    sorted_probs = jnp.sort(probs, axis=-1)[:, :, ::-1]
    cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
    mask_top_p = cumsum_probs < top_p
    mask_top_p = mask_top_p.at[:, :, -1].set(True)  # Keep at least one token
    
    # Compute entropy only for masked positions
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
    entropy = jnp.where(mask == 1, entropy, -jnp.inf)
    
    # Find positions with highest entropy (most uncertain)
    # This is where the model is least confident
    return entropy


def sample_tokens(
    logits: jnp.ndarray,
    rng: random.PRNGKey,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> jnp.ndarray:
    """Sample tokens from logits using top-p (nucleus) sampling.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        rng: JAX random key
        temperature: Sampling temperature
        top_p: Top-p threshold
        
    Returns:
        Sampled token indices [batch_size, seq_len]
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-p filtering
    probs = jax.nn.softmax(logits, axis=-1)
    sorted_indices = jnp.argsort(probs, axis=-1)[:, :, ::-1]
    sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
    cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
    
    # Create mask for top-p
    mask = cumsum_probs < top_p
    mask = mask.at[:, :, -1].set(True)  # Keep at least one token
    
    # Zero out probabilities outside top-p
    sorted_probs = jnp.where(mask, sorted_probs, 0.0)
    sorted_probs = sorted_probs / jnp.sum(sorted_probs, axis=-1, keepdims=True)
    
    # Sample from filtered distribution
    rng_batch = random.split(rng, batch_size * seq_len)
    rng_batch = rng_batch.reshape(batch_size, seq_len, -1)
    
    samples = jax.vmap(jax.vmap(lambda p, r: random.categorical(r, jnp.log(p + 1e-10))))(
        sorted_probs, rng_batch[:, :, 0]
    )
    
    # Convert back to original indices
    sampled_tokens = jnp.take_along_axis(
        sorted_indices, samples[:, :, None], axis=-1
    ).squeeze(-1)
    
    return sampled_tokens


@partial(jax.jit, static_argnames=["model", "num_steps", "tokens_per_step", "max_new_tokens"])
def diffusion_generate(
    model: DiffuCoder,
    params: Dict[str, Any],
    input_ids: jnp.ndarray,
    rng: random.PRNGKey,
    attention_mask: Optional[jnp.ndarray] = None,
    num_steps: int = 256,
    tokens_per_step: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """Generate text using masked diffusion process.
    
    Args:
        model: DiffuCoder model
        params: Model parameters
        input_ids: Input token IDs [batch_size, seq_len]
        rng: JAX random key
        attention_mask: Attention mask (optional)
        num_steps: Number of diffusion steps
        tokens_per_step: Tokens to unmask per step
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p threshold
        alg: Algorithm for choosing positions ("entropy" or "random")
        alg_temp: Temperature for position selection algorithm
        
    Returns:
        Dictionary with:
            - sequences: Generated sequences
            - scores: Generation scores
            - history: Generation history (if requested)
    """
    batch_size, prompt_len = input_ids.shape
    config = model.config
    
    # Prepare initial sequence with mask tokens
    total_len = prompt_len + max_new_tokens
    sequences = jnp.ones((batch_size, total_len), dtype=jnp.int32) * config.mask_token_id
    sequences = sequences.at[:, :prompt_len].set(input_ids)
    
    # Create mask (1 = masked, 0 = unmasked)
    mask = jnp.ones((batch_size, total_len), dtype=jnp.float32)
    mask = mask.at[:, :prompt_len].set(0)
    
    # Attention mask
    if attention_mask is None:
        attention_mask = jnp.ones((batch_size, prompt_len), dtype=jnp.float32)
    # Extend attention mask
    full_attention_mask = jnp.ones((batch_size, total_len), dtype=jnp.float32)
    full_attention_mask = full_attention_mask.at[:, :prompt_len].set(attention_mask)
    
    # History tracking
    history = []
    
    # Diffusion loop
    for step in range(num_steps):
        rng, step_rng = random.split(rng)
        
        # Forward pass
        outputs = model.apply(
            params,
            sequences,
            attention_mask=full_attention_mask,
            deterministic=True,
        )
        logits = outputs["logits"]
        
        # Sample tokens for masked positions
        rng, sample_rng = random.split(step_rng)
        sampled_tokens = sample_tokens(logits, sample_rng, temperature, top_p)
        
        # Update sequences with sampled tokens (only at masked positions)
        sequences = jnp.where(
            mask[:, :, None].astype(bool),
            sampled_tokens[:, :, None],
            sequences[:, :, None]
        ).squeeze(-1)
        
        # Determine which positions to unmask
        if alg == "entropy":
            # Use entropy to find most uncertain positions
            entropy_scores = compute_entropy_mask(logits, mask, alg_temp or temperature, top_p)
            
            # Select top tokens_per_step positions per sequence
            num_to_unmask = min(tokens_per_step, int(jnp.sum(mask)))
            if num_to_unmask > 0:
                # Get indices of highest entropy positions
                flat_entropy = entropy_scores.reshape(batch_size, -1)
                flat_mask = mask.reshape(batch_size, -1)
                
                # For each sequence, find top positions
                for b in range(batch_size):
                    seq_entropy = flat_entropy[b]
                    seq_mask = flat_mask[b]
                    
                    # Get masked positions sorted by entropy
                    masked_indices = jnp.where(seq_mask == 1)[0]
                    if len(masked_indices) > 0:
                        masked_entropy = seq_entropy[masked_indices]
                        sorted_indices = jnp.argsort(-masked_entropy)  # Descending
                        
                        # Unmask top positions
                        positions_to_unmask = masked_indices[sorted_indices[:num_to_unmask]]
                        mask = mask.at[b, positions_to_unmask].set(0)
        
        elif alg == "random":
            # Random unmasking
            rng, unmask_rng = random.split(step_rng)
            num_to_unmask = min(tokens_per_step, int(jnp.sum(mask)))
            
            if num_to_unmask > 0:
                # Randomly select positions to unmask
                masked_positions = jnp.where(mask.reshape(-1) == 1)[0]
                selected = random.choice(
                    unmask_rng,
                    masked_positions,
                    shape=(num_to_unmask,),
                    replace=False
                )
                
                # Convert flat indices back to 2D
                batch_indices = selected // total_len
                seq_indices = selected % total_len
                mask = mask.at[batch_indices, seq_indices].set(0)
        
        # Store history
        history.append({
            "sequences": sequences.copy(),
            "mask": mask.copy(),
            "step": step,
        })
        
        # Early stopping if all tokens are unmasked
        if jnp.sum(mask) == 0:
            break
    
    # Final forward pass to get scores
    outputs = model.apply(
        params,
        sequences,
        attention_mask=full_attention_mask,
        deterministic=True,
    )
    final_logits = outputs["logits"]
    
    # Compute sequence scores
    scores = jnp.sum(
        jax.nn.log_softmax(final_logits, axis=-1) * 
        jax.nn.one_hot(sequences, config.vocab_size + 2),
        axis=(1, 2)
    )
    
    return {
        "sequences": sequences,
        "scores": scores,
        "history": history,
        "attention_mask": full_attention_mask,
    }


def stream_diffusion_generate(
    model: DiffuCoder,
    params: Dict[str, Any],
    input_ids: jnp.ndarray,
    rng: random.PRNGKey,
    callback: Callable[[Dict[str, Any]], None],
    **kwargs,
):
    """Streaming version of diffusion generation with callback.
    
    This is useful for real-time applications where you want to
    see the generation progress.
    
    Args:
        model: DiffuCoder model
        params: Model parameters
        input_ids: Input token IDs
        rng: JAX random key
        callback: Function called after each step with current state
        **kwargs: Additional arguments passed to diffusion_generate
    """
    # For now, this is a wrapper that calls the callback with final result
    # In a full implementation, this would integrate with the generation loop
    result = diffusion_generate(model, params, input_ids, rng, **kwargs)
    
    # Call callback with final result
    callback({
        "sequences": result["sequences"],
        "scores": result["scores"],
        "done": True,
    })
    
    return result