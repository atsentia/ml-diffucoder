"""HuggingFace integration utilities for JAX models.

This module provides utilities for:
1. Converting between JAX pytrees and safetensors format
2. Sharding large models for efficient storage/loading
3. HuggingFace Hub integration
"""

import json
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import HfApi, hf_hub_download, upload_file
from safetensors import safe_open
from safetensors.flax import save_file, load_file
import orbax.checkpoint as ocp

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.utils.model_utils import load_config


# Constants
SHARD_SIZE_LIMIT = 5 * 1024 * 1024 * 1024  # 5GB per shard
SAFETENSORS_FORMAT = "safetensors"
ORBAX_FORMAT = "orbax"


def pytree_to_flat_dict(
    params: Dict[str, Any], 
    prefix: str = ""
) -> Dict[str, np.ndarray]:
    """Convert JAX pytree to flat dictionary suitable for safetensors.
    
    Args:
        params: JAX parameter pytree
        prefix: Prefix for parameter names
        
    Returns:
        Flat dictionary mapping parameter names to numpy arrays
    """
    # Flatten the nested pytree structure
    flat_params = flatten_dict(params, sep=".")
    
    # Convert to numpy arrays and rename to match HF conventions
    result = {}
    for key, value in flat_params.items():
        # Convert JAX array to numpy
        np_value = np.array(value)
        
        # Rename keys to match HuggingFace conventions
        # JAX uses 'kernel' for linear layers, PyTorch uses 'weight'
        hf_key = key.replace(".kernel", ".weight")
        
        # JAX Dense layers have transposed weights compared to PyTorch
        if ".kernel" in key and len(np_value.shape) == 2:
            np_value = np_value.T
            
        # Add prefix if specified
        if prefix:
            hf_key = f"{prefix}.{hf_key}"
            
        result[hf_key] = np_value
    
    return result


def flat_dict_to_pytree(
    flat_dict: Dict[str, np.ndarray],
    target_structure: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert flat dictionary back to JAX pytree structure.
    
    Args:
        flat_dict: Flat dictionary from safetensors
        target_structure: Target pytree structure (for shape reference)
        
    Returns:
        JAX parameter pytree
    """
    # First, rename keys from HF conventions back to JAX
    jax_dict = {}
    for key, value in flat_dict.items():
        # Convert weight -> kernel
        jax_key = key.replace(".weight", ".kernel")
        
        # Transpose linear layer weights
        if ".weight" in key and len(value.shape) == 2:
            value = value.T
            
        jax_dict[jax_key] = jnp.array(value)
    
    # Unflatten to match target structure
    # Split keys by '.' and rebuild nested structure
    result = {}
    for key, value in jax_dict.items():
        parts = key.split('.')
        current = result
        
        # Navigate/create nested structure
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value
        current[parts[-1]] = value
    
    return result


def calculate_shard_groups(
    params: Dict[str, Any],
    max_shard_size: int = SHARD_SIZE_LIMIT
) -> List[List[str]]:
    """Calculate how to group parameters into shards.
    
    Args:
        params: Flattened parameter dictionary
        max_shard_size: Maximum size per shard in bytes
        
    Returns:
        List of parameter groups (each group is a list of parameter names)
    """
    # Get sizes of all parameters
    param_sizes = {}
    for name, param in params.items():
        param_sizes[name] = param.nbytes
    
    # Group parameters into shards
    shards = []
    current_shard = []
    current_size = 0
    
    for name, size in sorted(param_sizes.items()):
        if current_size + size > max_shard_size and current_shard:
            # Start new shard
            shards.append(current_shard)
            current_shard = [name]
            current_size = size
        else:
            current_shard.append(name)
            current_size += size
    
    if current_shard:
        shards.append(current_shard)
    
    return shards


def save_sharded_safetensors(
    params: Dict[str, Any],
    save_path: Path,
    max_shard_size: int = SHARD_SIZE_LIMIT
) -> Dict[str, Any]:
    """Save JAX parameters as sharded safetensors files.
    
    Args:
        params: JAX parameter pytree
        save_path: Directory to save files
        max_shard_size: Maximum size per shard
        
    Returns:
        Index metadata for loading shards
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to flat dictionary
    flat_params = pytree_to_flat_dict(params)
    
    # Calculate sharding
    shard_groups = calculate_shard_groups(flat_params, max_shard_size)
    
    # Save shards and build index
    index = {
        "metadata": {
            "format": "safetensors",
            "total_size": sum(p.nbytes for p in flat_params.values())
        },
        "weight_map": {}
    }
    
    for i, param_names in enumerate(shard_groups):
        shard_name = f"model-{i+1:05d}-of-{len(shard_groups):05d}.safetensors"
        shard_path = save_path / shard_name
        
        # Extract parameters for this shard
        shard_params = {name: flat_params[name] for name in param_names}
        
        # Save shard
        save_file(shard_params, shard_path)
        
        # Update index
        for name in param_names:
            index["weight_map"][name] = shard_name
    
    # Save index
    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    
    return index


def load_sharded_safetensors(
    load_path: Path,
    target_structure: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load sharded safetensors files into JAX pytree.
    
    Args:
        load_path: Directory containing safetensors files
        target_structure: Target pytree structure (optional)
        
    Returns:
        JAX parameter pytree
    """
    load_path = Path(load_path)
    
    # Load index
    index_path = load_path / "model.safetensors.index.json"
    if not index_path.exists():
        # Try single file
        safetensors_files = list(load_path.glob("*.safetensors"))
        if len(safetensors_files) == 1:
            flat_params = load_file(safetensors_files[0])
            return flat_dict_to_pytree(flat_params, target_structure)
        else:
            raise ValueError(f"No index file found at {index_path}")
    
    with open(index_path, "r") as f:
        index = json.load(f)
    
    # Load all shards
    flat_params = {}
    loaded_shards = set()
    
    for param_name, shard_name in index["weight_map"].items():
        if shard_name not in loaded_shards:
            shard_path = load_path / shard_name
            shard_params = load_file(shard_path)
            flat_params.update(shard_params)
            loaded_shards.add(shard_name)
    
    # Convert back to pytree
    return flat_dict_to_pytree(flat_params, target_structure)


def save_model_for_huggingface(
    model: DiffuCoder,
    params: Dict[str, Any],
    save_path: Path,
    tokenizer_path: Optional[Path] = None,
    model_card: Optional[str] = None,
    use_safetensors: bool = True,
    max_shard_size: int = SHARD_SIZE_LIMIT
):
    """Save model in HuggingFace-compatible format.
    
    Args:
        model: DiffuCoder model instance
        params: Model parameters
        save_path: Directory to save model
        tokenizer_path: Path to tokenizer files (optional)
        model_card: Model card content (optional)
        use_safetensors: Whether to use safetensors format
        max_shard_size: Maximum shard size for safetensors
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = model.config.__dict__.copy()
    config_dict.update({
        "model_type": "dream",
        "architectures": ["DreamForCausalLM"],
        "framework": "jax"
    })
    
    with open(save_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save weights
    if use_safetensors:
        # Save as sharded safetensors
        save_sharded_safetensors(params, save_path, max_shard_size)
    else:
        # Save as pickle (single file)
        with open(save_path / "params.pkl", "wb") as f:
            pickle.dump(params, f)
    
    # Copy tokenizer files if provided
    if tokenizer_path:
        tokenizer_path = Path(tokenizer_path)
        tokenizer_files = [
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenization_dream.py"
        ]
        
        for file_name in tokenizer_files:
            src = tokenizer_path / file_name
            if src.exists():
                dst = save_path / file_name
                dst.write_text(src.read_text())
    
    # Save model card
    if model_card:
        with open(save_path / "README.md", "w") as f:
            f.write(model_card)
    
    print(f"Model saved to {save_path} in HuggingFace format")


def upload_to_huggingface(
    local_path: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False
) -> str:
    """Upload model to HuggingFace Hub.
    
    Args:
        local_path: Local directory containing model files
        repo_id: HuggingFace repository ID (username/model-name)
        token: HuggingFace API token (optional)
        private: Whether to create private repository
        create_pr: Whether to create a pull request
        
    Returns:
        URL of uploaded model
    """
    api = HfApi(token=token)
    local_path = Path(local_path)
    
    # Create repository if needed
    try:
        api.create_repo(repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload all files
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            
            print(f"Uploading {relative_path}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(relative_path),
                repo_id=repo_id,
                create_pr=create_pr
            )
    
    return f"https://huggingface.co/{repo_id}"


def load_from_huggingface(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    token: Optional[str] = None,
    dtype: Any = jnp.float32
) -> Tuple[DiffuCoder, Dict[str, Any]]:
    """Load model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        revision: Specific revision to load
        cache_dir: Local cache directory
        token: HuggingFace API token
        dtype: Data type for parameters
        
    Returns:
        Tuple of (model, params)
    """
    # Download config
    config_path = hf_hub_download(
        repo_id, 
        "config.json",
        revision=revision,
        cache_dir=cache_dir,
        token=token
    )
    
    config = load_config(Path(config_path))
    
    # Initialize model
    model = DiffuCoder(config, dtype=dtype)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    init_params = model.init(rng, dummy_input, deterministic=True)
    
    # Check for safetensors index
    try:
        index_path = hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            revision=revision,
            cache_dir=cache_dir,
            token=token
        )
        
        # Load sharded safetensors
        model_dir = Path(index_path).parent
        params = load_sharded_safetensors(model_dir, init_params)
        
    except Exception:
        # Try single file formats
        try:
            # Try safetensors
            st_path = hf_hub_download(
                repo_id,
                "model.safetensors",
                revision=revision,
                cache_dir=cache_dir,
                token=token
            )
            flat_params = load_file(st_path)
            params = flat_dict_to_pytree(flat_params, init_params)
            
        except Exception:
            # Fall back to pickle
            pkl_path = hf_hub_download(
                repo_id,
                "params.pkl",
                revision=revision,
                cache_dir=cache_dir,
                token=token
            )
            with open(pkl_path, "rb") as f:
                params = pickle.load(f)
    
    # Convert dtype if needed
    if dtype != jnp.float32:
        params = jax.tree_map(lambda x: x.astype(dtype), params)
    
    return model, params