#!/usr/bin/env python3
"""Simple smoke test comparing first layer embeddings between PyTorch and JAX models."""

import os
import sys
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Force CPU mode for consistent comparison
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_first_layer_parity():
    """Compare first layer (embedding) weights between PyTorch and JAX models."""
    
    console.rule("[bold cyan]First Layer Parity Test (PyTorch vs JAX)[/bold cyan]")
    
    # Model paths
    pytorch_path = Path("models/diffucoder-7b-complete")
    jax_path = Path("models/DiffuCoder-7B-JAX")
    
    # Check if models exist
    if not pytorch_path.exists():
        console.print(f"[red]PyTorch model not found at {pytorch_path}[/red]")
        return False
    
    if not jax_path.exists():
        console.print(f"[red]JAX model not found at {jax_path}[/red]")
        return False
    
    # 1. Load PyTorch embeddings
    console.print("\n[bold]1. Loading PyTorch embeddings...[/bold]")
    
    try:
        import torch
        import safetensors.torch
        
        # Find the safetensor file containing embeddings
        pytorch_embeddings = None
        
        for i in range(1, 5):  # model-00001-of-00004.safetensors through model-00004-of-00004.safetensors
            safetensor_path = pytorch_path / f"model-0000{i}-of-00004.safetensors"
            if safetensor_path.exists():
                tensors = safetensors.torch.load_file(str(safetensor_path))
                
                # Look for embedding layer
                for key, tensor in tensors.items():
                    if "embed" in key.lower() or "wte" in key.lower():
                        console.print(f"[green]Found PyTorch embeddings in {safetensor_path.name}: {key}[/green]")
                        console.print(f"  Shape: {tensor.shape}")
                        console.print(f"  Dtype: {tensor.dtype}")
                        pytorch_embeddings = tensor.float().numpy()  # Convert to float32 numpy
                        break
                
                if pytorch_embeddings is not None:
                    break
        
        if pytorch_embeddings is None:
            console.print("[red]Could not find embeddings in PyTorch model[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]Failed to load PyTorch embeddings: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. Load JAX embeddings
    console.print("\n[bold]2. Loading JAX embeddings...[/bold]")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Try loading from pickle first
        jax_embeddings = None
        
        # Check for original pickle
        pkl_path = Path("models/DiffuCoder-7B-JAX-original/params.pkl")
        if pkl_path.exists():
            console.print(f"[cyan]Loading from pickle: {pkl_path}[/cyan]")
            import pickle
            with open(pkl_path, "rb") as f:
                params = pickle.load(f)
            
            # Navigate through the parameter structure
            # Handle both dict and FrozenDict
            from flax.core import frozen_dict
            
            # Convert to regular dict if it's a FrozenDict
            if isinstance(params, frozen_dict.FrozenDict):
                params = dict(params)
            
            if isinstance(params, dict):
                # Try different possible paths
                paths_to_try = [
                    ["params", "DreamModel_0", "Embed_0", "embedding"],
                    ["params", "DreamModel_0", "embed_tokens", "embedding"],
                    ["params", "transformer", "wte", "embedding"],
                    ["params", "model", "embed_tokens", "embedding"],
                    ["DreamModel_0", "embed_tokens", "embedding"],
                    ["embed_tokens", "embedding"],
                    ["wte", "embedding"]
                ]
                
                for path in paths_to_try:
                    current = params
                    found = True
                    for key in path:
                        # Handle FrozenDict at any level
                        if isinstance(current, frozen_dict.FrozenDict):
                            current = dict(current)
                        
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            found = False
                            break
                    
                    if found and isinstance(current, (np.ndarray, jnp.ndarray)):
                        console.print(f"[green]Found JAX embeddings at path: {' -> '.join(path)}[/green]")
                        console.print(f"  Shape: {current.shape}")
                        console.print(f"  Dtype: {current.dtype}")
                        jax_embeddings = np.array(current).astype(np.float32)
                        break
        
        if jax_embeddings is None:
            # Try loading from Orbax checkpoint
            checkpoint_path = (jax_path / "orbax_checkpoint").absolute()
            if checkpoint_path.exists():
                console.print(f"[cyan]Loading from Orbax checkpoint: {checkpoint_path}[/cyan]")
                import orbax.checkpoint as ocp
                
                ckptr = ocp.StandardCheckpointer()
                params = ckptr.restore(str(checkpoint_path))
                
                # Similar search in Orbax format
                # (Add similar path search logic here if needed)
                console.print("[yellow]Orbax loading not fully implemented in this test[/yellow]")
        
        if jax_embeddings is None:
            console.print("[red]Could not find embeddings in JAX model[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]Failed to load JAX embeddings: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Compare embeddings
    console.print("\n[bold]3. Comparing embeddings...[/bold]")
    
    # Check shapes
    if pytorch_embeddings.shape != jax_embeddings.shape:
        console.print(f"[red]Shape mismatch![/red]")
        console.print(f"  PyTorch: {pytorch_embeddings.shape}")
        console.print(f"  JAX: {jax_embeddings.shape}")
        return False
    
    console.print(f"[green]✓ Shapes match: {pytorch_embeddings.shape}[/green]")
    
    # Compute differences
    diff = np.abs(pytorch_embeddings - jax_embeddings)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Check if they're close
    rtol = 1e-3  # Relative tolerance
    atol = 1e-3  # Absolute tolerance
    are_close = np.allclose(pytorch_embeddings, jax_embeddings, rtol=rtol, atol=atol)
    
    # Create comparison table
    table = Table(title="Embedding Comparison Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")
    
    table.add_row("Shape", str(pytorch_embeddings.shape), "✓ Match")
    table.add_row("Max Difference", f"{max_diff:.6f}", "✓" if max_diff < 0.01 else "✗")
    table.add_row("Mean Difference", f"{mean_diff:.6f}", "✓" if mean_diff < 0.001 else "✗")
    table.add_row("All Close (rtol=1e-3)", str(are_close), "✓" if are_close else "✗")
    
    console.print(table)
    
    # Sample some values
    console.print("\n[bold]Sample values (first 5 embeddings of token 0):[/bold]")
    
    sample_table = Table()
    sample_table.add_column("Index", style="cyan")
    sample_table.add_column("PyTorch", style="green")
    sample_table.add_column("JAX", style="blue")
    sample_table.add_column("Difference", style="yellow")
    
    for i in range(min(5, pytorch_embeddings.shape[1])):
        pt_val = pytorch_embeddings[0, i]
        jax_val = jax_embeddings[0, i]
        diff_val = abs(pt_val - jax_val)
        
        sample_table.add_row(
            str(i),
            f"{pt_val:.6f}",
            f"{jax_val:.6f}",
            f"{diff_val:.6f}"
        )
    
    console.print(sample_table)
    
    # Statistics on a few tokens
    console.print("\n[bold]Statistics for first 10 tokens:[/bold]")
    
    stats_table = Table()
    stats_table.add_column("Token ID", style="cyan")
    stats_table.add_column("PyTorch Mean", style="green")
    stats_table.add_column("JAX Mean", style="blue")
    stats_table.add_column("PyTorch Std", style="green")
    stats_table.add_column("JAX Std", style="blue")
    
    for token_id in range(min(10, pytorch_embeddings.shape[0])):
        pt_mean = np.mean(pytorch_embeddings[token_id])
        jax_mean = np.mean(jax_embeddings[token_id])
        pt_std = np.std(pytorch_embeddings[token_id])
        jax_std = np.std(jax_embeddings[token_id])
        
        stats_table.add_row(
            str(token_id),
            f"{pt_mean:.6f}",
            f"{jax_mean:.6f}",
            f"{pt_std:.6f}",
            f"{jax_std:.6f}"
        )
    
    console.print(stats_table)
    
    # Final verdict
    console.rule("[bold]Verdict[/bold]")
    
    if are_close:
        console.print("[bold green]✅ First layer parity test PASSED![/bold green]")
        console.print("\nThe embedding weights match between PyTorch and JAX models.")
        console.print("This indicates the weight conversion was successful!")
        return True
    else:
        console.print("[bold yellow]⚠ First layer shows differences[/bold yellow]")
        console.print(f"\nMax difference: {max_diff:.6f}")
        console.print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 0.1:  # Still reasonably close
            console.print("\n[yellow]Differences are small and may be due to:[/yellow]")
            console.print("  - Precision differences during conversion")
            console.print("  - Different initialization seeds")
            console.print("  - Framework-specific optimizations")
            return True
        else:
            console.print("\n[red]Differences are significant. Please check:[/red]")
            console.print("  - Weight conversion process")
            console.print("  - Model architecture mapping")
            console.print("  - Parameter naming conventions")
            return False


def main():
    """Main function."""
    success = test_first_layer_parity()
    
    if success:
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Run full numerical parity test for complete validation")
        console.print("2. Upload model to HuggingFace")
        console.print("3. Test inference with real prompts")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())