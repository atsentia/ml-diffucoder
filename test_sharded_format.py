#!/usr/bin/env python3
"""Test script to verify JAX sharded model format."""

import json
import pickle
from pathlib import Path
import sys
import os

# Set JAX to use CPU for testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def test_pickle_format(model_dir: Path):
    """Test loading standard pickle format."""
    console.print(f"\n[bold blue]Testing pickle format in {model_dir}[/bold blue]")
    
    params_path = model_dir / "params.pkl"
    config_path = model_dir / "config.json"
    
    # Check files exist
    if not params_path.exists():
        console.print(f"[red]✗ params.pkl not found[/red]")
        return False
    else:
        console.print(f"[green]✓ params.pkl found ({params_path.stat().st_size / 1e9:.2f} GB)[/green]")
    
    if not config_path.exists():
        console.print(f"[red]✗ config.json not found[/red]")
        return False
    else:
        console.print(f"[green]✓ config.json found[/green]")
    
    # Try loading params
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading parameters...", total=None)
            
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            
            progress.update(task, completed=True)
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        console.print(f"[green]✓ Successfully loaded {param_count:,} parameters[/green]")
        
        # Show parameter structure
        table = Table(title="Parameter Structure")
        table.add_column("Layer", style="cyan")
        table.add_column("Shape", style="magenta")
        table.add_column("Size (MB)", style="green")
        
        for key, value in jax.tree_util.tree_flatten_with_path(params)[0][:5]:
            path = ".".join(str(k.key) if hasattr(k, 'key') else str(k) for k in key)
            shape = str(value.shape)
            size_mb = value.nbytes / 1e6 if hasattr(value, 'nbytes') else value.size * 4 / 1e6
            table.add_row(path, shape, f"{size_mb:.1f}")
        
        if len(jax.tree_util.tree_leaves(params)) > 5:
            table.add_row("...", "...", "...")
        
        console.print(table)
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Failed to load parameters: {e}[/red]")
        return False


def test_orbax_format(model_dir: Path):
    """Test loading Orbax sharded format."""
    console.print(f"\n[bold blue]Testing Orbax format in {model_dir}[/bold blue]")
    
    checkpoint_dir = model_dir / "orbax_checkpoint"
    metadata_path = model_dir / "checkpoint_metadata.json"
    config_path = model_dir / "config.json"
    
    # Check files exist
    if not checkpoint_dir.exists():
        console.print(f"[red]✗ orbax_checkpoint directory not found[/red]")
        return False
    else:
        console.print(f"[green]✓ orbax_checkpoint directory found[/green]")
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        console.print(f"[green]✓ Metadata found: {metadata.get('num_shards', 'unknown')} shards[/green]")
    
    # List checkpoint contents
    checkpoint_files = list(checkpoint_dir.rglob("*"))
    console.print(f"[cyan]Found {len(checkpoint_files)} files in checkpoint[/cyan]")
    
    # Try loading checkpoint
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading Orbax checkpoint...", total=None)
            
            # Try different loading methods
            ckptr = ocp.PyTreeCheckpointer()
            
            # First try standard restore
            try:
                params = ckptr.restore(checkpoint_dir)
                loading_method = "standard"
            except:
                # Try with checkpoint manager
                manager = ocp.CheckpointManager(
                    checkpoint_dir.parent,
                    options=ocp.CheckpointManagerOptions(create=False),
                    item_names=("params",),
                )
                restored = manager.restore(manager.latest_step())
                params = restored.get("params", restored)
                loading_method = "manager"
            
            progress.update(task, completed=True)
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        console.print(f"[green]✓ Successfully loaded {param_count:,} parameters using {loading_method} method[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Failed to load Orbax checkpoint: {e}[/red]")
        return False


def test_model_loading(model_dir: Path):
    """Test loading model through high-level API."""
    console.print(f"\n[bold blue]Testing high-level model loading from {model_dir}[/bold blue]")
    
    try:
        # Add jax_lm to path
        jax_lm_path = Path(__file__).parent / "jax_lm"
        if jax_lm_path.exists():
            sys.path.insert(0, str(jax_lm_path.parent))
        
        from jax_lm import load_model
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading model via high-level API...", total=None)
            
            model, params = load_model(str(model_dir))
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✓ Successfully loaded model via high-level API[/green]")
        console.print(f"[cyan]Model type: {type(model).__name__}[/cyan]")
        
        return True
        
    except Exception as e:
        console.print(f"[yellow]⚠ Could not test high-level API: {e}[/yellow]")
        return None


def main():
    """Test different model formats."""
    console.print("[bold]JAX Model Format Tester[/bold]\n")
    
    # Test directories
    test_dirs = [
        ("Original JAX model", Path("models/dream-jax")),
        ("Sharded JAX model", Path("models/dream-jax-sharded")),
    ]
    
    results = []
    
    for name, path in test_dirs:
        if not path.exists():
            console.print(f"[yellow]⚠ {name} not found at {path}[/yellow]")
            continue
        
        console.rule(name)
        
        # Detect format
        if (path / "params.pkl").exists():
            result = test_pickle_format(path)
        elif (path / "orbax_checkpoint").exists():
            result = test_orbax_format(path)
        else:
            console.print(f"[red]✗ Unknown format in {path}[/red]")
            result = False
        
        # Test high-level loading if basic test passed
        if result:
            api_result = test_model_loading(path)
            if api_result is not None:
                result = result and api_result
        
        results.append((name, result))
    
    # Summary
    console.rule("Summary")
    
    summary_table = Table(title="Test Results")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Status", style="green")
    
    for name, result in results:
        status = "[green]✓ PASSED[/green]" if result else "[red]✗ FAILED[/red]"
        summary_table.add_row(name, status)
    
    console.print(summary_table)
    
    # Overall result
    all_passed = all(r for _, r in results)
    if all_passed:
        console.print("\n[bold green]✅ All tests passed![/bold green]")
        return 0
    else:
        console.print("\n[bold red]❌ Some tests failed![/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())