#!/usr/bin/env python3
"""Numerical parity test between PyTorch and JAX DiffuCoder models."""

import os
import sys
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Force CPU mode for consistent comparison
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_ENABLE_X64"] = "0"  # Use float32 for consistency


def load_pytorch_model(model_path):
    """Load PyTorch model and tokenizer."""
    try:
        import torch
        import sys
        
        console.print(f"[cyan]Loading PyTorch model from {model_path}[/cyan]")
        
        # Add model path to sys.path so we can import custom modules
        sys.path.insert(0, str(model_path))
        
        # Import custom Dream model classes
        from modeling_dream import DreamModel
        from configuration_dream import DreamConfig
        from tokenization_dream import DreamTokenizer
        
        # Load config
        config = DreamConfig.from_pretrained(model_path)
        
        # Load model in float32 for consistency
        model = DreamModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model.eval()
        
        # Load tokenizer
        tokenizer = DreamTokenizer.from_pretrained(model_path)
        
        # Remove from path
        sys.path.remove(str(model_path))
        
        return model, tokenizer
        
    except Exception as e:
        console.print(f"[red]Failed to load PyTorch model: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None, None


def load_jax_model(model_path):
    """Load JAX model and tokenizer."""
    try:
        import jax
        import jax.numpy as jnp
        import json
        import orbax.checkpoint as ocp
        from jax_lm.models.dream import DreamForCausalLM, DreamConfig
        from jax_lm.utils.tokenizer import DreamTokenizer
        
        console.print(f"[cyan]Loading JAX model from {model_path}[/cyan]")
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        config = DreamConfig(**config_dict)
        
        # Initialize model
        model = DreamForCausalLM(config)
        
        # Load tokenizer
        tokenizer = DreamTokenizer.from_pretrained(str(model_path))
        
        # Create dummy input for initialization
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        
        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, dummy_input)
        
        # Load weights from checkpoint
        ckptr = ocp.StandardCheckpointer()
        checkpoint_path = model_path / "orbax_checkpoint"
        
        if checkpoint_path.exists():
            console.print(f"[cyan]Loading from Orbax checkpoint: {checkpoint_path}[/cyan]")
            params = ckptr.restore(checkpoint_path, target=params)
        else:
            # Try loading from pickle file if available
            import pickle
            pkl_path = model_path.parent / "DiffuCoder-7B-JAX-original" / "params.pkl"
            if pkl_path.exists():
                console.print(f"[cyan]Loading from pickle: {pkl_path}[/cyan]")
                with open(pkl_path, "rb") as f:
                    params = pickle.load(f)
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or {pkl_path}")
        
        return model, params, tokenizer
        
    except Exception as e:
        console.print(f"[red]Failed to load JAX model: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_pytorch_outputs(model, tokenizer, prompt):
    """Get outputs from PyTorch model."""
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        
        # Extract key tensors
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        return {
            "logits": logits.cpu().numpy(),
            "hidden_states": hidden_states.cpu().numpy(),
            "input_ids": input_ids.cpu().numpy()
        }


def get_jax_outputs(model, params, tokenizer, prompt):
    """Get outputs from JAX model."""
    import jax
    import jax.numpy as jnp
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    input_ids = jnp.array(inputs["input_ids"])
    
    # Forward pass
    outputs = model.apply(
        params,
        input_ids,
        deterministic=True,
        output_hidden_states=True
    )
    
    # Extract outputs
    return {
        "logits": np.array(outputs.logits),
        "hidden_states": np.array(outputs.hidden_states[-1]),
        "input_ids": np.array(input_ids)
    }


def compare_outputs(pytorch_outputs, jax_outputs, tolerance=1e-3):
    """Compare PyTorch and JAX outputs."""
    results = {}
    
    # Compare logits
    logits_diff = np.abs(pytorch_outputs["logits"] - jax_outputs["logits"])
    logits_max_diff = np.max(logits_diff)
    logits_mean_diff = np.mean(logits_diff)
    logits_close = np.allclose(
        pytorch_outputs["logits"], 
        jax_outputs["logits"], 
        rtol=tolerance, 
        atol=tolerance
    )
    
    results["logits"] = {
        "max_diff": float(logits_max_diff),
        "mean_diff": float(logits_mean_diff),
        "close": logits_close,
        "pytorch_shape": pytorch_outputs["logits"].shape,
        "jax_shape": jax_outputs["logits"].shape
    }
    
    # Compare hidden states
    hidden_diff = np.abs(pytorch_outputs["hidden_states"] - jax_outputs["hidden_states"])
    hidden_max_diff = np.max(hidden_diff)
    hidden_mean_diff = np.mean(hidden_diff)
    hidden_close = np.allclose(
        pytorch_outputs["hidden_states"], 
        jax_outputs["hidden_states"], 
        rtol=tolerance, 
        atol=tolerance
    )
    
    results["hidden_states"] = {
        "max_diff": float(hidden_max_diff),
        "mean_diff": float(hidden_mean_diff),
        "close": hidden_close,
        "pytorch_shape": pytorch_outputs["hidden_states"].shape,
        "jax_shape": jax_outputs["hidden_states"].shape
    }
    
    # Sample some actual values for inspection
    results["sample_logits"] = {
        "pytorch": pytorch_outputs["logits"][0, -1, :5].tolist(),
        "jax": jax_outputs["logits"][0, -1, :5].tolist()
    }
    
    return results


def main():
    """Run numerical parity test."""
    console.rule("[bold cyan]PyTorch vs JAX Numerical Parity Test[/bold cyan]")
    
    # Model paths
    pytorch_path = Path("models/diffucoder-7b-complete")
    jax_path = Path("models/DiffuCoder-7B-JAX")
    
    # Check paths
    if not pytorch_path.exists():
        console.print(f"[red]PyTorch model not found at {pytorch_path}[/red]")
        console.print("[yellow]Please download the PyTorch model first[/yellow]")
        return 1
    
    if not jax_path.exists():
        console.print(f"[red]JAX model not found at {jax_path}[/red]")
        console.print("[yellow]Please convert the model first[/yellow]")
        return 1
    
    # Test prompts
    test_prompts = [
        "def hello_world():",
        "class Calculator:\n    def __init__(self):",
        "# Function to compute factorial\ndef factorial(n):"
    ]
    
    # Load models
    console.print("\n[bold]Loading models...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Load PyTorch
        task = progress.add_task("Loading PyTorch model...", total=None)
        pytorch_model, pytorch_tokenizer = load_pytorch_model(pytorch_path)
        progress.update(task, completed=True)
        
        if pytorch_model is None:
            console.print("[red]Failed to load PyTorch model[/red]")
            return 1
        
        console.print("[green]✓ PyTorch model loaded[/green]")
        
        # Load JAX
        task = progress.add_task("Loading JAX model...", total=None)
        jax_model, jax_params, jax_tokenizer = load_jax_model(jax_path)
        progress.update(task, completed=True)
        
        if jax_model is None:
            console.print("[red]Failed to load JAX model[/red]")
            return 1
        
        console.print("[green]✓ JAX model loaded[/green]")
    
    # Run comparisons
    console.print("\n[bold]Running numerical comparisons...[/bold]")
    
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Comparing outputs...", total=len(test_prompts))
        
        for i, prompt in enumerate(test_prompts):
            # Get outputs
            pytorch_out = get_pytorch_outputs(pytorch_model, pytorch_tokenizer, prompt)
            jax_out = get_jax_outputs(jax_model, jax_params, jax_tokenizer, prompt)
            
            # Compare
            results = compare_outputs(pytorch_out, jax_out)
            all_results[prompt] = results
            
            progress.advance(task)
    
    # Display results
    console.rule("[bold]Results[/bold]")
    
    # Summary table
    summary_table = Table(title="Numerical Comparison Summary")
    summary_table.add_column("Prompt", style="cyan", width=30)
    summary_table.add_column("Component", style="magenta")
    summary_table.add_column("Max Diff", style="yellow")
    summary_table.add_column("Mean Diff", style="yellow")
    summary_table.add_column("Status", style="green")
    
    all_close = True
    
    for prompt, results in all_results.items():
        prompt_short = prompt.split('\n')[0][:30] + "..."
        
        # Logits row
        logits_result = results["logits"]
        status = "✓ PASS" if logits_result["close"] else "✗ FAIL"
        if not logits_result["close"]:
            all_close = False
            
        summary_table.add_row(
            prompt_short,
            "Logits",
            f"{logits_result['max_diff']:.6f}",
            f"{logits_result['mean_diff']:.6f}",
            status
        )
        
        # Hidden states row
        hidden_result = results["hidden_states"]
        status = "✓ PASS" if hidden_result["close"] else "✗ FAIL"
        if not hidden_result["close"]:
            all_close = False
            
        summary_table.add_row(
            "",
            "Hidden States",
            f"{hidden_result['max_diff']:.6f}",
            f"{hidden_result['mean_diff']:.6f}",
            status
        )
    
    console.print(summary_table)
    
    # Sample values
    console.print("\n[bold]Sample Logit Values (first 5):[/bold]")
    sample_table = Table()
    sample_table.add_column("Prompt", style="cyan", width=30)
    sample_table.add_column("PyTorch", style="green")
    sample_table.add_column("JAX", style="blue")
    
    for prompt, results in all_results.items():
        prompt_short = prompt.split('\n')[0][:30] + "..."
        pytorch_vals = [f"{v:.4f}" for v in results["sample_logits"]["pytorch"]]
        jax_vals = [f"{v:.4f}" for v in results["sample_logits"]["jax"]]
        
        sample_table.add_row(
            prompt_short,
            str(pytorch_vals),
            str(jax_vals)
        )
    
    console.print(sample_table)
    
    # Final verdict
    console.rule("[bold]Verdict[/bold]")
    
    if all_close:
        console.print("[bold green]✅ Numerical parity PASSED![/bold green]")
        console.print("\nThe JAX model produces outputs that are numerically")
        console.print("equivalent to the PyTorch model within tolerance.")
        console.print("\n[green]The model is ready for deployment![/green]")
    else:
        console.print("[bold yellow]⚠ Numerical differences detected[/bold yellow]")
        console.print("\nThe models show some numerical differences.")
        console.print("This may be due to:")
        console.print("  - Precision differences (float32 vs mixed precision)")
        console.print("  - Implementation differences in ops")
        console.print("  - Weight conversion issues")
        console.print("\n[yellow]Further investigation recommended[/yellow]")
    
    return 0 if all_close else 1


if __name__ == "__main__":
    sys.exit(main())