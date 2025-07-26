"""JAX-DiffuCoder: High-performance JAX/Flax implementation of DiffuCoder.

A 7.6B parameter diffusion-based code generation model optimized for TPUs and GPUs.
"""

__version__ = "0.1.0"

# Core imports
from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.models import diffusion_generate, batch_diffusion_generate

# Pure JAX loader (no HuggingFace dependencies)
from jax_lm.utils.pure_jax_loader import (
    load_model,
    load_tokenizer,
    PureJAXModelLoader
)

# Model utilities
from jax_lm.utils.model_utils import (
    initialize_model,
    save_model,
    count_parameters,
    parameter_summary
)

# Orbax sharding utilities
from jax_lm.utils.orbax_sharding import (
    ShardedCheckpointer,
    save_for_huggingface
)

# Tokenizer
from jax_lm.utils.tokenizer import DreamTokenizer

# Convenience function for generation
def generate(
    model,
    params,
    prompt: str,
    tokenizer=None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42
):
    """Convenience function for text generation.
    
    Args:
        model: DiffuCoder model instance
        params: Model parameters
        prompt: Input prompt string
        tokenizer: Tokenizer instance (will load default if None)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        seed: Random seed
        
    Returns:
        Generated text string
    """
    import jax
    import jax.numpy as jnp
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = load_tokenizer("atsentia/DiffuCoder-7B-JAX")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    
    # Generate
    output = diffusion_generate(
        model,
        params,
        input_ids,
        jax.random.PRNGKey(seed),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    # Decode
    generated_ids = output["sequences"][0]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def batch_generate(
    model,
    params,
    prompts: list,
    tokenizer=None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42
):
    """Convenience function for batch generation.
    
    Args:
        model: DiffuCoder model instance
        params: Model parameters
        prompts: List of input prompts
        tokenizer: Tokenizer instance
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        seed: Random seed
        
    Returns:
        List of generated texts
    """
    import jax
    import jax.numpy as jnp
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = load_tokenizer("atsentia/DiffuCoder-7B-JAX")
    
    # Tokenize prompts
    inputs = tokenizer(prompts, padding=True, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    attention_mask = jnp.array(inputs["attention_mask"])
    
    # Generate
    outputs = batch_diffusion_generate(
        model,
        params,
        input_ids,
        jax.random.PRNGKey(seed),
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    # Decode all outputs
    results = []
    for seq in outputs["sequences"]:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        results.append(text)
    
    return results


__all__ = [
    # Core classes
    "DiffuCoder",
    "DiffuCoderConfig",
    "DreamTokenizer",
    "PureJAXModelLoader",
    "ShardedCheckpointer",
    
    # Main functions
    "load_model",
    "load_tokenizer",
    "generate",
    "batch_generate",
    "diffusion_generate",
    "batch_diffusion_generate",
    
    # Utilities
    "initialize_model",
    "save_model",
    "save_for_huggingface",
    "count_parameters",
    "parameter_summary",
]