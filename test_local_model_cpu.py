#!/usr/bin/env python3
"""Test JAX model loading and inference on CPU."""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import time
from pathlib import Path
import pickle
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


def test_local_model():
    """Test loading and running the local JAX model on CPU."""
    
    console.rule("[bold cyan]JAX Model CPU Inference Test[/bold cyan]")
    
    # Model paths
    model_dir = Path("/Users/amund/ml-diffucoder/models/DiffuCoder-7B-JAX")
    original_params = Path("/Users/amund/ml-diffucoder/models/DiffuCoder-7B-JAX-original/params.pkl")
    
    if not model_dir.exists():
        console.print(f"[red]Model directory not found: {model_dir}[/red]")
        return False
    
    console.print(f"[green]✓ Found model directory: {model_dir}[/green]")
    
    # 1. Load config
    console.print("\n[bold]1. Loading configuration...[/bold]")
    
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)
    
    console.print(f"[green]✓ Model type: {config.get('model_type', 'unknown')}[/green]")
    console.print(f"[green]✓ Hidden size: {config.get('hidden_size', 'unknown')}[/green]")
    console.print(f"[green]✓ Layers: {config.get('num_hidden_layers', 'unknown')}[/green]")
    
    # 2. Load parameters
    console.print("\n[bold]2. Loading model parameters...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        if original_params.exists():
            task = progress.add_task("Loading from pickle file...", total=None)
            
            start_time = time.time()
            with open(original_params, "rb") as f:
                params = pickle.load(f)
            load_time = time.time() - start_time
            
            progress.update(task, completed=True)
            console.print(f"[green]✓ Loaded parameters in {load_time:.2f}s[/green]")
            
            # Analyze params
            from flax.core import frozen_dict
            if isinstance(params, frozen_dict.FrozenDict):
                params = params.unfreeze()
            
            if 'params' in params and 'DreamModel_0' in params['params']:
                model_params = params['params']['DreamModel_0']
                console.print(f"[green]✓ Found DreamModel parameters[/green]")
                
                # Count parameters
                import jax
                param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
                console.print(f"[green]✓ Total parameters: {param_count:,}[/green]")
                
                # Show sample parameter shapes
                param_table = Table(title="Sample Parameter Shapes")
                param_table.add_column("Parameter", style="cyan")
                param_table.add_column("Shape", style="green")
                param_table.add_column("Size (MB)", style="yellow")
                
                def add_params_to_table(obj, prefix="", depth=0, max_depth=2):
                    if depth > max_depth:
                        return
                    
                    if isinstance(obj, dict):
                        for key, value in list(obj.items())[:3]:  # First 3 items
                            new_prefix = f"{prefix}.{key}" if prefix else key
                            if hasattr(value, 'shape'):
                                size_mb = value.size * value.dtype.itemsize / 1e6
                                param_table.add_row(new_prefix, str(value.shape), f"{size_mb:.1f}")
                            elif isinstance(value, dict):
                                add_params_to_table(value, new_prefix, depth + 1, max_depth)
                
                add_params_to_table(model_params)
                console.print(param_table)
                
            else:
                console.print("[yellow]⚠ Unexpected parameter structure[/yellow]")
                console.print(f"Top-level keys: {list(params.keys())}")
                if 'params' in params:
                    console.print(f"Params keys: {list(params['params'].keys())}")
        
        else:
            console.print("[red]✗ Original params.pkl not found[/red]")
            return False
    
    # 3. Test simple forward pass
    console.print("\n[bold]3. Testing model initialization...[/bold]")
    
    try:
        # Add jax_lm to path
        sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))
        
        # Try importing
        from models.diffucoder import DiffuCoder, DiffuCoderConfig
        
        # Create config
        model_config = DiffuCoderConfig(**config)
        
        # Initialize model
        import jax
        import jax.numpy as jnp
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing model...", total=None)
            
            model = DiffuCoder(model_config)
            
            # Create dummy input
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
            
            # Initialize
            init_vars = model.init(rng, dummy_input, deterministic=True)
            
            progress.update(task, completed=True)
        
        console.print("[green]✓ Model initialized successfully[/green]")
        
        # 4. Test inference
        console.print("\n[bold]4. Testing inference (this will be slow on CPU)...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running forward pass...", total=None)
            
            start_time = time.time()
            
            # Run forward pass
            outputs = model.apply(
                {'params': params['params']},
                dummy_input,
                deterministic=True
            )
            
            inference_time = time.time() - start_time
            progress.update(task, completed=True)
        
        console.print(f"[green]✓ Forward pass completed in {inference_time:.2f}s[/green]")
        console.print(f"[green]✓ Output shape: {outputs.shape}[/green]")
        
        # Summary
        console.rule("[bold]Summary[/bold]")
        console.print("[bold green]✅ All tests passed![/bold green]")
        console.print("\nModel is ready for:")
        console.print("  1. Upload to HuggingFace Hub")
        console.print("  2. PyPI package publication")
        console.print("  3. Testing on TPU/GPU")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = test_local_model()
    
    if success:
        console.print("\n[cyan]Note: CPU inference is very slow for a 7.6B parameter model.[/cyan]")
        console.print("[cyan]TPU/GPU will provide much better performance.[/cyan]")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())