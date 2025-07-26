"""Orbax-based sharding utilities for JAX models.

This module provides utilities for saving and loading large JAX models
using Orbax's native sharding capabilities, optimized for HuggingFace Hub.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint as ocp
from orbax.checkpoint import type_handlers
import tensorstore as ts

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.utils.model_utils import load_config, initialize_model


# Constants
SHARD_SIZE_LIMIT = 5 * 1024 * 1024 * 1024  # 5GB per shard
CHECKPOINT_DIR = "orbax_checkpoint"
METADATA_FILE = "checkpoint_metadata.json"


class ShardedCheckpointer:
    """Handles sharded saving/loading of JAX models using Orbax."""
    
    def __init__(
        self,
        max_shard_size: int = SHARD_SIZE_LIMIT,
        use_zarr: bool = True
    ):
        """Initialize the sharded checkpointer.
        
        Args:
            max_shard_size: Maximum size per shard in bytes
            use_zarr: Whether to use Zarr format for better sharding
        """
        self.max_shard_size = max_shard_size
        self.use_zarr = use_zarr
        
        # Configure Orbax options for sharding
        self.save_args = ocp.SaveArgs(
            aggregate=True,  # Enable aggregation for better performance
        )
        
    def estimate_param_sizes(self, params: Dict[str, Any]) -> Dict[str, int]:
        """Estimate sizes of parameters in bytes."""
        sizes = {}
        flat_params = flatten_dict(params)
        
        for key, value in flat_params.items():
            if hasattr(value, 'nbytes'):
                sizes[key] = value.nbytes
            else:
                # Estimate size based on shape and dtype
                shape = value.shape if hasattr(value, 'shape') else ()
                dtype_size = np.dtype(value.dtype).itemsize if hasattr(value, 'dtype') else 4
                size = np.prod(shape) * dtype_size if shape else dtype_size
                sizes[key] = size
        
        return sizes
    
    def create_sharding_spec(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a sharding specification for the parameters.
        
        This determines how parameters should be split across files.
        """
        param_sizes = self.estimate_param_sizes(params)
        total_size = sum(param_sizes.values())
        
        # Create sharding metadata
        sharding_spec = {
            "total_size": total_size,
            "num_shards": max(1, (total_size + self.max_shard_size - 1) // self.max_shard_size),
            "shard_size_limit": self.max_shard_size,
            "format": "orbax_zarr" if self.use_zarr else "orbax_standard",
            "parameters": {}
        }
        
        # Group parameters by layer for better organization
        layer_groups = {}
        for param_path, size in param_sizes.items():
            # Extract layer name (e.g., "layer_0", "embedding", etc.)
            parts = param_path.split('.')
            layer_name = parts[0] if parts else "misc"
            
            if layer_name not in layer_groups:
                layer_groups[layer_name] = []
            layer_groups[layer_name].append((param_path, size))
        
        # Assign parameters to shards
        current_shard = 0
        current_size = 0
        
        for layer_name, layer_params in sorted(layer_groups.items()):
            for param_path, size in layer_params:
                if current_size + size > self.max_shard_size and current_size > 0:
                    current_shard += 1
                    current_size = 0
                
                sharding_spec["parameters"][param_path] = {
                    "shard": current_shard,
                    "size": size,
                    "layer": layer_name
                }
                current_size += size
        
        return sharding_spec
    
    def save_sharded(
        self,
        params: Dict[str, Any],
        save_path: Path,
        config: Optional[DiffuCoderConfig] = None
    ) -> Dict[str, Any]:
        """Save parameters with sharding using Orbax.
        
        Args:
            params: Model parameters
            save_path: Directory to save the checkpoint
            config: Model configuration (optional)
            
        Returns:
            Metadata about the saved checkpoint
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create sharding specification
        sharding_spec = self.create_sharding_spec(params)
        
        # Configure Orbax for sharding
        if self.use_zarr:
            # Use Zarr-based storage for automatic sharding
            manager = ocp.CheckpointManager(
                save_path / CHECKPOINT_DIR,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=1,
                    max_to_keep=1,
                    create=True,
                ),
                item_names=("params",),
            )
            
            # Save with automatic sharding
            manager.save(
                0,  # step
                args=ocp.args.Composite(
                    params=ocp.args.JsonSave(params)
                    if sharding_spec["num_shards"] == 1
                    else ocp.args.PyTreeSave(
                        params,
                        aggregate_filename="params.zarr",
                        ocdbt_target_data_file_size=self.max_shard_size,
                    )
                )
            )
            
            # Wait for save to complete
            manager.wait_until_finished()
            
        else:
            # Use standard Orbax checkpointing
            ckptr = ocp.PyTreeCheckpointer()
            ckptr.save(save_path / CHECKPOINT_DIR, params, save_args=self.save_args)
        
        # Save configuration if provided
        if config:
            config_dict = config.__dict__.copy()
            config_dict.update({
                "model_type": "diffucoder",
                "architectures": ["DiffuCoderForCausalLM"],
                "framework": "jax",
                "checkpoint_format": "orbax_sharded"
            })
            
            with open(save_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
        
        # Save checkpoint metadata
        metadata = {
            "format": "orbax_sharded",
            "sharding": sharding_spec,
            "checkpoint_dir": CHECKPOINT_DIR,
            "model_type": "diffucoder",
            "framework": "jax",
            "orbax_version": ocp.__version__,
            "jax_version": jax.__version__,
        }
        
        with open(save_path / METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved sharded checkpoint to {save_path}")
        print(f"   Total size: {sharding_spec['total_size'] / 1e9:.2f} GB")
        print(f"   Number of shards: {sharding_spec['num_shards']}")
        
        return metadata
    
    def load_sharded(
        self,
        load_path: Path,
        target_structure: Optional[Dict[str, Any]] = None,
        dtype: Any = jnp.float32
    ) -> Dict[str, Any]:
        """Load sharded parameters using Orbax.
        
        Args:
            load_path: Directory containing the checkpoint
            target_structure: Target parameter structure (optional)
            dtype: Data type to cast parameters to
            
        Returns:
            Loaded parameters
        """
        load_path = Path(load_path)
        
        # Load metadata
        metadata_path = load_path / METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"format": "orbax_standard"}
        
        # Load checkpoint
        checkpoint_path = load_path / CHECKPOINT_DIR
        
        if metadata.get("format") == "orbax_sharded" and self.use_zarr:
            # Load Zarr-based checkpoint
            manager = ocp.CheckpointManager(
                checkpoint_path.parent,
                options=ocp.CheckpointManagerOptions(create=False),
                item_names=("params",),
            )
            
            # Restore from latest checkpoint
            restored = manager.restore(
                manager.latest_step(),
                args=ocp.args.Composite(
                    params=ocp.args.PyTreeRestore(
                        target_structure or {},
                        restore_kwargs={"dtype": dtype}
                    )
                )
            )
            params = restored["params"]
            
        else:
            # Load standard checkpoint
            ckptr = ocp.PyTreeCheckpointer()
            
            if target_structure:
                params = ckptr.restore(checkpoint_path, target=target_structure)
            else:
                params = ckptr.restore(checkpoint_path)
        
        # Convert dtype if needed
        if dtype != jnp.float32:
            params = jax.tree_map(lambda x: x.astype(dtype), params)
        
        return params


def save_for_huggingface(
    model: DiffuCoder,
    params: Dict[str, Any],
    save_path: Path,
    tokenizer_path: Optional[Path] = None,
    model_card: Optional[str] = None,
    max_shard_size: int = SHARD_SIZE_LIMIT
):
    """Save model in HuggingFace-compatible format using Orbax sharding.
    
    Args:
        model: DiffuCoder model instance
        params: Model parameters
        save_path: Directory to save model
        tokenizer_path: Path to tokenizer files (optional)
        model_card: Model card content (optional)
        max_shard_size: Maximum shard size in bytes
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize sharded checkpointer
    checkpointer = ShardedCheckpointer(max_shard_size=max_shard_size)
    
    # Save sharded checkpoint
    metadata = checkpointer.save_sharded(params, save_path, model.config)
    
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
                shutil.copy2(src, dst)
    
    # Save model card
    if model_card:
        with open(save_path / "README.md", "w") as f:
            f.write(model_card)
    
    # Create a loading script for convenience
    loading_script = '''"""Load this model using the jax_lm library."""

from jax_lm.utils.orbax_sharding import load_from_huggingface

# Load the model
model, params = load_from_huggingface("{repo_id}")

# Use the model
from jax_lm.models import diffusion_generate
import jax

output = diffusion_generate(
    model,
    params, 
    input_ids,
    jax.random.PRNGKey(0),
    max_new_tokens=256
)
'''
    
    with open(save_path / "loading_instructions.py", "w") as f:
        f.write(loading_script)
    
    print(f"Model saved in HuggingFace-compatible Orbax format at {save_path}")


def load_from_huggingface(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    token: Optional[str] = None,
    dtype: Any = jnp.float32
) -> Tuple[DiffuCoder, Dict[str, Any]]:
    """Load model from HuggingFace Hub using Orbax format.
    
    Args:
        repo_id: HuggingFace repository ID  
        revision: Specific revision to load
        cache_dir: Local cache directory
        token: HuggingFace API token
        dtype: Data type for parameters
        
    Returns:
        Tuple of (model, params)
    """
    from huggingface_hub import snapshot_download
    
    # Download all files
    local_dir = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token
    )
    
    local_path = Path(local_dir)
    
    # Load config
    config = load_config(local_path / "config.json")
    
    # Initialize model structure
    model = DiffuCoder(config, dtype=dtype)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    init_params = model.init(rng, dummy_input, deterministic=True)
    
    # Check for different checkpoint formats
    if (local_path / "params.pkl").exists():
        # Load legacy pickle format
        import pickle
        print("Loading from pickle format...")
        with open(local_path / "params.pkl", "rb") as f:
            params = pickle.load(f)
    elif (local_path / CHECKPOINT_DIR).exists():
        # Load sharded Orbax checkpoint
        print("Loading from Orbax sharded format...")
        checkpointer = ShardedCheckpointer()
        params = checkpointer.load_sharded(local_path, init_params, dtype)
    else:
        raise ValueError(f"No valid checkpoint found in {local_path}")
    
    # Convert dtype if needed
    if dtype != jnp.float32:
        params = jax.tree_map(lambda x: x.astype(dtype), params)
    
    return model, params