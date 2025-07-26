"""JAX-DiffuCoder: High-performance JAX/Flax implementation of DiffuCoder.

A 7.6B parameter diffusion-based code generation model optimized for TPUs and GPUs.
"""

__version__ = "0.1.2"

# Define exports
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

# Simple convenience functions that don't cause circular imports
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
    
    # Lazy imports to avoid circular dependency
    from .utils.pure_jax_loader import load_tokenizer as _load_tokenizer
    from .models.diffucoder import diffusion_generate as _diffusion_generate
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = _load_tokenizer("atsentia/DiffuCoder-7B-JAX")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    
    # Generate
    output = _diffusion_generate(
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
    
    # Lazy imports
    from .utils.pure_jax_loader import load_tokenizer as _load_tokenizer
    from .models.diffucoder import batch_diffusion_generate as _batch_diffusion_generate
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = _load_tokenizer("atsentia/DiffuCoder-7B-JAX")
    
    # Tokenize prompts
    inputs = tokenizer(prompts, padding=True, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    attention_mask = jnp.array(inputs["attention_mask"])
    
    # Generate
    outputs = _batch_diffusion_generate(
        model, params, input_ids,
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


# Lazy loading functions for other imports
def load_model(*args, **kwargs):
    """Load model from path or HuggingFace Hub."""
    from .utils.pure_jax_loader import load_model as _load_model
    return _load_model(*args, **kwargs)


def load_tokenizer(*args, **kwargs):
    """Load tokenizer."""
    from .utils.pure_jax_loader import load_tokenizer as _load_tokenizer
    return _load_tokenizer(*args, **kwargs)


# The actual imports will happen when these are accessed
def __getattr__(name):
    """Lazy loading of heavy imports."""
    if name == "DiffuCoder":
        from .models.diffucoder import DiffuCoder
        return DiffuCoder
    elif name == "DiffuCoderConfig":
        from .models.diffucoder import DiffuCoderConfig
        return DiffuCoderConfig
    elif name == "diffusion_generate":
        from .models.diffucoder import diffusion_generate
        return diffusion_generate
    elif name == "batch_diffusion_generate":
        from .models.diffucoder import batch_diffusion_generate
        return batch_diffusion_generate
    elif name == "DreamTokenizer":
        from .utils.tokenizer import DreamTokenizer
        return DreamTokenizer
    elif name == "PureJAXModelLoader":
        from .utils.pure_jax_loader import PureJAXModelLoader
        return PureJAXModelLoader
    elif name == "ShardedCheckpointer":
        from .utils.orbax_sharding import ShardedCheckpointer
        return ShardedCheckpointer
    elif name == "initialize_model":
        from .utils.model_utils import initialize_model
        return initialize_model
    elif name == "save_model":
        from .utils.model_utils import save_model
        return save_model
    elif name == "save_for_huggingface":
        from .utils.orbax_sharding import save_for_huggingface
        return save_for_huggingface
    elif name == "count_parameters":
        from .utils.model_utils import count_parameters
        return count_parameters
    elif name == "parameter_summary":
        from .utils.model_utils import parameter_summary
        return parameter_summary
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")