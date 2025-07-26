#!/usr/bin/env python3
"""Inspect JAX model parameters structure."""

import pickle
import json
from pathlib import Path
from rich.console import Console
from rich.tree import Tree
import jax

console = Console()

def print_param_tree(params, max_depth=3):
    """Print parameter tree structure."""
    
    def add_to_tree(node, obj, depth=0):
        if depth > max_depth:
            return
        
        if isinstance(obj, dict):
            for key, value in sorted(obj.items())[:10]:  # Show first 10 keys
                if isinstance(value, dict):
                    sub_node = node.add(f"[bold cyan]{key}[/bold cyan] (dict)")
                    add_to_tree(sub_node, value, depth + 1)
                elif hasattr(value, 'shape'):
                    node.add(f"[green]{key}[/green] {value.shape} {value.dtype}")
                else:
                    node.add(f"[yellow]{key}[/yellow] ({type(value).__name__})")
            if len(obj) > 10:
                node.add("... and more")
        elif hasattr(obj, 'shape'):
            node.add(f"Array {obj.shape} {obj.dtype}")
        else:
            node.add(f"{type(obj).__name__}")
    
    tree = Tree("Parameters")
    add_to_tree(tree, params)
    return tree


def main():
    """Inspect JAX model parameters."""
    
    console.rule("[bold cyan]JAX Model Parameter Inspector[/bold cyan]")
    
    # Load original params
    original_pkl = Path("models/DiffuCoder-7B-JAX-original/params.pkl")
    if original_pkl.exists():
        console.print(f"\n[bold]Loading original params from:[/bold] {original_pkl}")
        
        with open(original_pkl, "rb") as f:
            params = pickle.load(f)
        
        console.print("\n[bold]Parameter Structure:[/bold]")
        tree = print_param_tree(params)
        console.print(tree)
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        console.print(f"\n[bold]Total parameters:[/bold] {param_count:,}")
        
        # Get first level keys
        if isinstance(params, dict):
            console.print(f"\n[bold]Top-level keys:[/bold] {list(params.keys())}")
            
            # Look for embeddings
            console.print("\n[bold]Searching for embeddings...[/bold]")
            
            def find_embeddings(obj, path=""):
                results = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        if 'embed' in key.lower() or 'wte' in key.lower():
                            if hasattr(value, 'shape'):
                                results.append((new_path, value.shape))
                            else:
                                results.append((new_path, type(value).__name__))
                        results.extend(find_embeddings(value, new_path))
                return results
            
            embeddings = find_embeddings(params)
            if embeddings:
                console.print("[green]Found embeddings:[/green]")
                for path, shape in embeddings[:5]:
                    console.print(f"  {path}: {shape}")
            else:
                console.print("[yellow]No embeddings found[/yellow]")
    
    # Also check Orbax checkpoint
    orbax_path = Path("models/DiffuCoder-7B-JAX/orbax_checkpoint")
    if orbax_path.exists():
        console.print(f"\n[bold]Orbax checkpoint exists at:[/bold] {orbax_path}")
        
        # List files
        files = list(orbax_path.rglob("*"))[:20]
        console.print(f"Sample files:")
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                console.print(f"  {f.relative_to(orbax_path)}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()