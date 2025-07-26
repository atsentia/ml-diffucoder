#!/usr/bin/env python3
"""Convert existing JAX model to sharded Orbax format for HuggingFace upload.

This script:
1. Loads the existing params.pkl file
2. Converts it to sharded Orbax format (4 shards ~5GB each)
3. Prepares for HuggingFace upload
"""

import argparse
import json
import pickle
from pathlib import Path
import shutil
import sys

import jax
import jax.numpy as jnp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use absolute imports
import utils.orbax_sharding as orbax_sharding
import utils.model_utils as model_utils
import models.diffucoder as diffucoder

ShardedCheckpointer = orbax_sharding.ShardedCheckpointer
save_for_huggingface = orbax_sharding.save_for_huggingface
load_config = model_utils.load_config
count_parameters = model_utils.count_parameters
DiffuCoder = diffucoder.DiffuCoder
DiffuCoderConfig = diffucoder.DiffuCoderConfig


def main():
    parser = argparse.ArgumentParser(
        description="Convert JAX model to sharded format"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../../models/dream-jax"),
        help="Input directory with params.pkl and config.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../models/dream-jax-sharded"),
        help="Output directory for sharded model"
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("../../models/diffucoder-7b-complete"),
        help="Directory with tokenizer files"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Target number of shards (default: 4)"
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="Maximum size per shard"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for parameters"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.tokenizer_dir = args.tokenizer_dir.resolve()
    
    print(f"Converting model from {args.input_dir} to sharded format...")
    
    # Check input files exist
    params_path = args.input_dir / "params.pkl"
    config_path = args.input_dir / "config.json"
    
    if not params_path.exists():
        print(f"Error: params.pkl not found at {params_path}")
        return 1
    
    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        return 1
    
    # Load config
    print("Loading configuration...")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = DiffuCoderConfig(**config_dict)
    
    # Load parameters
    print("Loading parameters...")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    # Count parameters
    param_count = count_parameters(params)
    print(f"Model has {param_count:,} parameters")
    
    # Calculate total size
    total_bytes = sum(
        x.nbytes if hasattr(x, 'nbytes') else x.size * 4
        for x in jax.tree_leaves(params)
    )
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    
    # Parse shard size
    shard_size_str = args.max_shard_size.upper()
    if shard_size_str.endswith("GB"):
        max_shard_bytes = int(float(shard_size_str[:-2]) * 1024 * 1024 * 1024)
    elif shard_size_str.endswith("MB"):
        max_shard_bytes = int(float(shard_size_str[:-2]) * 1024 * 1024)
    else:
        max_shard_bytes = int(shard_size_str)
    
    # Adjust shard size to match target number of shards
    target_shard_size = total_bytes // args.num_shards
    if target_shard_size > max_shard_bytes:
        print(f"Warning: Target shard size {target_shard_size/1e9:.2f}GB exceeds max {max_shard_bytes/1e9:.2f}GB")
        print(f"Will create more than {args.num_shards} shards")
    else:
        max_shard_bytes = int(target_shard_size * 1.1)  # Add 10% margin
        print(f"Using shard size: {max_shard_bytes/1e9:.2f}GB to create ~{args.num_shards} shards")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert dtype if needed
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16
    }
    target_dtype = dtype_map[args.dtype]
    
    if args.dtype != "float32":
        print(f"Converting to {args.dtype}...")
        params = jax.tree_map(lambda x: x.astype(target_dtype), params)
    
    # Initialize model (needed for save_for_huggingface)
    print("Initializing model structure...")
    model = DiffuCoder(config, dtype=target_dtype)
    
    # Create model card
    model_card = f"""---
license: apple-ascl
language:
- code
library_name: jax
tags:
- diffusion
- code-generation
- jax
- flax
- tpu
- orbax
datasets:
- TIGER-Lab/AceCode-89K
base_model: apple/DiffuCoder-7B-Instruct
---

# DiffuCoder-7B-JAX (Sharded)

A JAX/Flax implementation of DiffuCoder with sharded checkpoint format for efficient loading.

This model uses Orbax sharding with {args.num_shards} shards for optimal loading performance.

## Usage

```python
from jax_lm import load_model, generate

# Load sharded model
model, params = load_model("atsentia/DiffuCoder-7B-JAX")

# Generate code
output = generate(model, params, "def fibonacci(n):", max_new_tokens=256)
print(output)
```

See the [main repository](https://github.com/atsentia/ml-diffucoder) for more information.
"""
    
    # Save in sharded format
    print(f"\nSaving to sharded format at {args.output_dir}...")
    save_for_huggingface(
        model,
        params,
        args.output_dir,
        tokenizer_path=args.tokenizer_dir,
        model_card=model_card,
        max_shard_size=max_shard_bytes
    )
    
    # List created files
    print("\nCreated files:")
    total_size = 0
    for file_path in sorted(args.output_dir.rglob("*")):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            rel_path = file_path.relative_to(args.output_dir)
            print(f"  {rel_path}: {size / 1e6:.1f} MB")
    
    print(f"\nTotal size: {total_size / 1e9:.2f} GB")
    print(f"✅ Conversion complete! Model saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())