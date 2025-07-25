"""Model loading and saving utilities."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import orbax.checkpoint as ocp

from jax_lm.models.dream import DreamForCausalLM as DiffuCoder
from jax_lm.models.dream import DreamConfig as DiffuCoderConfig


def load_config(config_path: Path) -> DiffuCoderConfig:
    """Load model configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return DiffuCoderConfig(**config_dict)


def save_config(config: DiffuCoderConfig, save_path: Path):
    """Save model configuration to JSON file."""
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)


def initialize_model(
    config: DiffuCoderConfig,
    rng: jax.random.PRNGKey,
    dtype: Any = jnp.float32,
) -> Tuple[DiffuCoder, Dict[str, Any]]:
    """Initialize a DiffuCoder model with random parameters.
    
    Args:
        config: Model configuration
        rng: Random key for initialization
        dtype: Data type for model parameters
        
    Returns:
        Tuple of (model, params)
    """
    model = DiffuCoder(config, dtype=dtype)
    
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    
    # Initialize parameters
    params = model.init(rng, dummy_input, deterministic=True)
    
    return model, params


def load_model(
    model_path: str,
    dtype: Any = jnp.float32,
    device: Optional[str] = None,
) -> Tuple[DiffuCoder, Dict[str, Any]]:
    """Load a DiffuCoder model from disk.
    
    Args:
        model_path: Path to model directory or HuggingFace model ID
        dtype: Data type for model parameters
        device: Device to load model on (optional)
        
    Returns:
        Tuple of (model, params)
    """
    model_path = Path(model_path)
    
    # Check if this is a local path or HF model ID
    if model_path.exists():
        # Local path
        config_path = model_path / "config.json"
        params_path = model_path / "params.pkl"
        
        # Check for pickle format first (converted weights)
        if not params_path.exists():
            # Try checkpoint format
            checkpoint_path = model_path / "checkpoint"
            if checkpoint_path.exists():
                params_path = checkpoint_path
                use_pickle = False
            else:
                raise FileNotFoundError(
                    f"No params.pkl or checkpoint found in {model_path}"
                )
        else:
            use_pickle = True
    else:
        # Assume HuggingFace model ID
        # In production, this would download from HF
        raise NotImplementedError(
            "Loading from HuggingFace not yet implemented. "
            "Please download the model first using download_weights.py"
        )
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    model, _ = initialize_model(config, rng, dtype)
    
    # Load parameters
    if use_pickle:
        # Load from pickle format (converted weights)
        import pickle
        with open(params_path, "rb") as f:
            params = pickle.load(f)
    else:
        # Load from orbax checkpoint
        ckptr = ocp.PyTreeCheckpointer()
        params = ckptr.restore(params_path)
    
    # Convert to specified dtype if needed
    if dtype != jnp.float32:
        params = jax.tree_map(lambda x: x.astype(dtype), params)
    
    return model, params


def save_model(
    model: DiffuCoder,
    params: Dict[str, Any],
    save_path: Path,
):
    """Save a DiffuCoder model to disk.
    
    Args:
        model: DiffuCoder model instance
        params: Model parameters
        save_path: Directory to save model
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(model.config, save_path)
    
    # Save parameters
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(save_path / "checkpoint", params)
    
    print(f"Model saved to {save_path}")


def convert_pytorch_checkpoint(
    pytorch_path: str,
    output_path: str,
    dtype: Any = jnp.float32,
) -> Tuple[DiffuCoder, Dict[str, Any]]:
    """Convert a PyTorch checkpoint to JAX format.
    
    This is a wrapper around the conversion script functionality.
    
    Args:
        pytorch_path: Path to PyTorch checkpoint
        output_path: Path to save JAX checkpoint
        dtype: Data type for model parameters
        
    Returns:
        Tuple of (model, params)
    """
    # Import conversion utilities
    from jax_lm.scripts.convert_pytorch_to_jax import convert_pytorch_to_jax
    
    # Run conversion
    pytorch_path = Path(pytorch_path)
    output_path = Path(output_path)
    
    convert_pytorch_to_jax(pytorch_path, output_path)
    
    # Load the converted model
    return load_model(output_path, dtype)


def count_parameters(params: Dict[str, Any]) -> int:
    """Count the total number of parameters in a parameter tree."""
    return sum(x.size for x in jax.tree_leaves(params))


def parameter_summary(params: Dict[str, Any]) -> Dict[str, int]:
    """Get a summary of parameters by layer type."""
    summary = {}
    
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    
    for path, param in flat_params:
        # Extract layer type from path
        path_str = "/".join(str(p) for p in path)
        
        if "embedding" in path_str.lower():
            layer_type = "embeddings"
        elif "attention" in path_str.lower():
            layer_type = "attention"
        elif "mlp" in path_str.lower():
            layer_type = "mlp"
        elif "norm" in path_str.lower():
            layer_type = "normalization"
        elif "lm_head" in path_str.lower():
            layer_type = "lm_head"
        else:
            layer_type = "other"
        
        if layer_type not in summary:
            summary[layer_type] = 0
        summary[layer_type] += param.size
    
    summary["total"] = sum(summary.values())
    
    return summary