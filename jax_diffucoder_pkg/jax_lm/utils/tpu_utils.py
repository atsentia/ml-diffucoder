"""TPU-specific utilities for JAX DiffuCoder."""

from typing import Dict, Any, Optional, Tuple
import os

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding


def setup_tpu():
    """Initialize TPU environment."""
    # Set up JAX for TPU
    if "TPU_NAME" in os.environ:
        # Running on Cloud TPU
        import requests
        
        # Get TPU metadata
        tpu_name = os.environ["TPU_NAME"]
        zone = os.environ.get("TPU_ZONE", "us-central1-a")
        project = os.environ.get("GCP_PROJECT", "")
        
        print(f"Connecting to TPU: {tpu_name} in zone {zone}")
        
        # TPU initialization is handled automatically by JAX
        devices = jax.devices()
        print(f"Found {len(devices)} TPU devices")
        
        return devices
    else:
        # Local mode or CPU/GPU
        devices = jax.devices()
        device_type = devices[0].device_kind
        print(f"Running on {len(devices)} {device_type} device(s)")
        return devices


def get_tpu_mesh(devices: Optional[list] = None) -> Mesh:
    """Create a mesh for data parallelism across TPU devices.
    
    Args:
        devices: List of devices (if None, uses all available)
        
    Returns:
        JAX Mesh object for sharding
    """
    if devices is None:
        devices = jax.devices()
    
    # Create mesh with data parallelism
    # For a TPU v4-8, this would be shape (8,)
    mesh_shape = (len(devices),)
    devices_array = mesh_utils.create_device_mesh(mesh_shape)
    
    # Create mesh with axis names
    mesh = Mesh(devices_array, axis_names=("data",))
    
    print(f"Created mesh with shape {mesh_shape}")
    
    return mesh


def shard_params(
    params: Dict[str, Any],
    mesh: Mesh,
    model_parallel_size: int = 1,
) -> Dict[str, Any]:
    """Shard model parameters across TPU devices.
    
    Args:
        params: Model parameters to shard
        mesh: JAX mesh object
        model_parallel_size: Number of devices for model parallelism
        
    Returns:
        Sharded parameters
    """
    # Define sharding specs for different parameter types
    def get_partition_spec(path: Tuple[str, ...], value: jnp.ndarray) -> PartitionSpec:
        """Determine partition spec based on parameter path and shape."""
        path_str = "/".join(path)
        
        # Don't shard small parameters
        if value.size < 1024:
            return PartitionSpec()
        
        # Embedding layers - shard along vocab dimension if large
        if "embedding" in path_str.lower():
            if value.ndim == 2 and value.shape[0] > 10000:
                return PartitionSpec("data", None)
            return PartitionSpec()
        
        # Attention parameters - potential for model parallelism
        if "attention" in path_str.lower():
            if model_parallel_size > 1:
                # Shard attention heads across model parallel dimension
                if "q_proj" in path_str or "k_proj" in path_str or "v_proj" in path_str:
                    if value.ndim == 2:
                        return PartitionSpec(None, "model")
                elif "o_proj" in path_str:
                    if value.ndim == 2:
                        return PartitionSpec("model", None)
            return PartitionSpec()
        
        # MLP parameters
        if "mlp" in path_str.lower():
            if model_parallel_size > 1:
                if "gate_proj" in path_str or "up_proj" in path_str:
                    if value.ndim == 2:
                        return PartitionSpec(None, "model")
                elif "down_proj" in path_str:
                    if value.ndim == 2:
                        return PartitionSpec("model", None)
            return PartitionSpec()
        
        # Default: replicate across all devices
        return PartitionSpec()
    
    # Flatten parameters
    flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params)
    
    # Create sharding for each parameter
    sharded_params = []
    for path, value in flat_params:
        partition_spec = get_partition_spec(path, value)
        sharding = NamedSharding(mesh, partition_spec)
        sharded_value = jax.device_put(value, sharding)
        sharded_params.append(sharded_value)
    
    # Reconstruct tree
    return jax.tree_util.tree_unflatten(tree_def, sharded_params)


def profile_tpu_memory(
    fn,
    *args,
    **kwargs,
):
    """Profile TPU memory usage for a function.
    
    Args:
        fn: Function to profile
        *args: Positional arguments for fn
        **kwargs: Keyword arguments for fn
        
    Returns:
        Function output and memory profile
    """
    # Run function
    output = fn(*args, **kwargs)
    
    # Get memory stats
    devices = jax.devices()
    memory_stats = []
    
    for device in devices:
        stats = device.memory_stats()
        if stats:
            memory_stats.append({
                "device": str(device),
                "bytes_in_use": stats.get("bytes_in_use", 0),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                "bytes_limit": stats.get("bytes_limit", 0),
            })
    
    return output, memory_stats


def optimize_for_tpu(
    model_fn,
    static_argnums: Tuple[int, ...] = (),
    donate_argnums: Tuple[int, ...] = (),
):
    """Optimize a model function for TPU execution.
    
    Args:
        model_fn: Model function to optimize
        static_argnums: Indices of static arguments
        donate_argnums: Indices of arguments to donate (for memory efficiency)
        
    Returns:
        Optimized function
    """
    # Apply XLA flags for TPU optimization
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    
    # JIT compile with TPU-specific options
    optimized_fn = jax.jit(
        model_fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )
    
    return optimized_fn