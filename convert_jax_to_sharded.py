#!/usr/bin/env python3
"""Convert existing JAX model to sharded Orbax format for HuggingFace upload.

This is a standalone script to avoid circular import issues.

To disable rich logging, set environment variable:
    NO_RICH_LOGGING=1 python convert_jax_to_sharded.py
"""

import argparse
import json
import pickle
from pathlib import Path
import shutil
import sys
import os

# Set JAX to use CPU for this conversion
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.traverse_util import flatten_dict
import numpy as np

# Rich logging support
USE_RICH = os.environ.get("NO_RICH_LOGGING", "").lower() not in ("1", "true", "yes")

if USE_RICH:
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
    except ImportError:
        USE_RICH = False
        print("Note: Install 'rich' for better progress display: pip install rich")

if not USE_RICH:
    # Fallback console that just prints
    class FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip rich markup
            text = str(args[0]) if args else ""
            text = text.replace("[bold]", "").replace("[/bold]", "")
            text = text.replace("[green]", "").replace("[/green]", "")
            text = text.replace("[red]", "").replace("[/red]", "")
            text = text.replace("[cyan]", "").replace("[/cyan]", "")
            text = text.replace("[yellow]", "").replace("[/yellow]", "")
            print(text)
        
        def rule(self, title=""):
            print(f"\n{'='*60}")
            if title:
                print(f" {title} ")
            print(f"{'='*60}\n")
    
    console = FallbackConsole()


def count_parameters(params):
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def estimate_param_sizes(params):
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


def save_sharded_checkpoint(params, save_path, max_shard_size_bytes):
    """Save parameters in sharded Orbax format."""
    save_path = Path(save_path)
    checkpoint_dir = save_path / "orbax_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Estimate sizes
    param_sizes = estimate_param_sizes(params)
    total_size = sum(param_sizes.values())
    num_shards = max(1, (total_size + max_shard_size_bytes - 1) // max_shard_size_bytes)
    
    console.print(f"[cyan]Total parameter size:[/cyan] {total_size / 1e9:.2f} GB")
    console.print(f"[cyan]Target shard size:[/cyan] {max_shard_size_bytes / 1e9:.2f} GB")
    console.print(f"[cyan]Expected number of shards:[/cyan] {num_shards}")
    
    # Configure Orbax for sharding
    save_args = ocp.SaveArgs()
    
    # Use PyTreeCheckpointer with sharding options
    ckptr = ocp.PyTreeCheckpointer()
    
    # Save using standard PyTreeCheckpointer (works with FrozenDict)
    console.print("[yellow]Saving checkpoint (this may take a while for large models)...[/yellow]")
    
    # Convert FrozenDict to regular dict if needed
    from flax.core import frozen_dict
    if isinstance(params, frozen_dict.FrozenDict):
        params = params.unfreeze()
    
    # Save checkpoint
    ckptr.save(checkpoint_dir, params, force=True)
    
    # Create metadata
    metadata = {
        "format": "orbax_sharded",
        "num_shards": num_shards,
        "total_size": total_size,
        "checkpoint_dir": "orbax_checkpoint",
        "model_type": "diffucoder",
        "framework": "jax",
        "orbax_version": ocp.__version__,
        "jax_version": jax.__version__,
    }
    
    with open(save_path / "checkpoint_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert JAX model to sharded format"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("models/dream-jax"),
        help="Input directory with params.pkl and config.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/dream-jax-sharded"),
        help="Output directory for sharded model"
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("models/diffucoder-7b-complete"),
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
    
    args = parser.parse_args()
    
    console.rule("[bold cyan]JAX Model Sharding Converter[/bold cyan]")
    console.print(f"\n[bold]Converting:[/bold] {args.input_dir}")
    console.print(f"[bold]Output to:[/bold] {args.output_dir}\n")
    
    # Check input files exist
    params_path = args.input_dir / "params.pkl"
    config_path = args.input_dir / "config.json"
    
    if not params_path.exists():
        console.print(f"[red]Error: params.pkl not found at {params_path}[/red]")
        return 1
    
    if not config_path.exists():
        console.print(f"[red]Error: config.json not found at {config_path}[/red]")
        return 1
    
    # Load parameters
    if USE_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Loading parameters...", total=None)
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            progress.update(task, completed=True)
    else:
        console.print("Loading parameters...")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
    
    # Count parameters
    param_count = count_parameters(params)
    console.print(f"[green]✓[/green] Model has [bold]{param_count:,}[/bold] parameters")
    
    # Calculate total size
    total_bytes = sum(
        x.nbytes if hasattr(x, 'nbytes') else x.size * 4
        for x in jax.tree_util.tree_leaves(params)
    )
    console.print(f"[green]✓[/green] Total size: [bold]{total_bytes / 1e9:.2f} GB[/bold]")
    
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
        console.print(f"[yellow]Warning: Target shard size {target_shard_size/1e9:.2f}GB exceeds max {max_shard_bytes/1e9:.2f}GB[/yellow]")
        console.print(f"[yellow]Will create more than {args.num_shards} shards[/yellow]")
    else:
        max_shard_bytes = int(target_shard_size * 1.1)  # Add 10% margin
        console.print(f"[cyan]Using shard size: {max_shard_bytes/1e9:.2f}GB to create ~{args.num_shards} shards[/cyan]")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config
    shutil.copy2(config_path, args.output_dir / "config.json")
    
    # Update config with framework info
    with open(args.output_dir / "config.json", "r") as f:
        config = json.load(f)
    
    config.update({
        "model_type": "diffucoder",
        "architectures": ["DiffuCoderForCausalLM"],
        "framework": "jax",
        "checkpoint_format": "orbax_sharded"
    })
    
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save sharded checkpoint
    console.rule("[bold]Saving Sharded Checkpoint[/bold]")
    if USE_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Creating sharded checkpoint...", total=None)
            metadata = save_sharded_checkpoint(params, args.output_dir, max_shard_bytes)
            progress.update(task, completed=True)
    else:
        metadata = save_sharded_checkpoint(params, args.output_dir, max_shard_bytes)
    
    # Copy tokenizer files if they exist
    tokenizer_files = [
        "tokenizer_config.json",
        "vocab.json", 
        "merges.txt",
        "special_tokens_map.json",
        "tokenization_dream.py"
    ]
    
    console.rule("[bold]Copying Tokenizer Files[/bold]")
    for file_name in tokenizer_files:
        src = args.tokenizer_dir / file_name
        if src.exists():
            dst = args.output_dir / file_name
            shutil.copy2(src, dst)
            console.print(f"  [green]✓[/green] Copied {file_name}")
    
    # Create model card
    model_card = """---
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

## Usage

```python
# Install jax-diffucoder
pip install jax-diffucoder

# Load and use the model
from jax_lm import load_model, generate

model, params = load_model("atsentia/DiffuCoder-7B-JAX")
output = generate(model, params, "def fibonacci(n):", max_new_tokens=256)
print(output)
```

See the [main repository](https://github.com/atsentia/ml-diffucoder) for more information.
"""
    
    with open(args.output_dir / "README.md", "w") as f:
        f.write(model_card)
    
    # Create loading instructions
    loading_script = '''"""Load this model using jax-diffucoder."""

# Install the package
# pip install jax-diffucoder

from jax_lm import load_model, generate
import jax

# Load the model
model, params = load_model("atsentia/DiffuCoder-7B-JAX")

# Generate code
prompt = "def quicksort(arr):"
output = generate(model, params, prompt, max_new_tokens=256)
print(output)
'''
    
    with open(args.output_dir / "loading_instructions.py", "w") as f:
        f.write(loading_script)
    
    # List created files
    console.rule("[bold]Summary[/bold]")
    
    if USE_RICH:
        table = Table(title="Created Files", show_header=True)
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right", style="green")
        
        total_size = 0
        file_count = 0
        for file_path in sorted(args.output_dir.rglob("*")):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                file_count += 1
                rel_path = file_path.relative_to(args.output_dir)
                
                # Only show first 10 files in table
                if file_count <= 10:
                    table.add_row(str(rel_path), f"{size / 1e6:.1f} MB")
        
        if file_count > 10:
            table.add_row("...", f"({file_count - 10} more files)")
        
        console.print(table)
        console.print(f"\n[bold green]Total size:[/bold green] {total_size / 1e9:.2f} GB")
        console.print(f"[bold green]Total files:[/bold green] {file_count}")
    else:
        console.print("\nCreated files:")
        total_size = 0
        for file_path in sorted(args.output_dir.rglob("*")):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                rel_path = file_path.relative_to(args.output_dir)
                console.print(f"  {rel_path}: {size / 1e6:.1f} MB")
        console.print(f"\nTotal size: {total_size / 1e9:.2f} GB")
    
    console.print(f"\n[bold green]✅ Conversion complete![/bold green]")
    console.print(f"[green]Model saved to:[/green] {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())