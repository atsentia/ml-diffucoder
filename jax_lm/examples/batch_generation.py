#!/usr/bin/env python3
"""Batch generation example for JAX DiffuCoder."""

import jax
import jax.numpy as jnp
from jax_lm import load_model, diffusion_generate
from jax_lm.utils import load_tokenizer, prepare_input_ids


def batch_generate(model, params, tokenizer, prompts, rng, **kwargs):
    """Generate completions for multiple prompts efficiently."""
    
    # Tokenize all prompts
    inputs = prepare_input_ids(
        tokenizer,
        prompts,
        padding=True,
        truncation=True,
        return_tensors="jax",
    )
    
    # Generate for batch
    output = diffusion_generate(
        model,
        params,
        inputs["input_ids"],
        rng,
        attention_mask=inputs["attention_mask"],
        **kwargs
    )
    
    # Decode each sequence
    results = []
    for i, seq in enumerate(output["sequences"]):
        # Find actual length (before padding)
        actual_length = jnp.sum(inputs["attention_mask"][i])
        generated_text = tokenizer.decode(seq[:actual_length], skip_special_tokens=True)
        results.append(generated_text)
    
    return results, output


def main():
    # Setup
    model_path = "./models/jax"
    
    print("Loading model and tokenizer...")
    model, params = load_model(model_path, dtype=jnp.bfloat16)
    tokenizer = load_tokenizer(f"{model_path}/tokenizer")
    
    # Multiple prompts for batch processing
    prompts = [
        "def merge_sort(arr):",
        "class LinkedList:\n    def __init__(self):",
        "function validateEmail(email) {",
        "SELECT * FROM users WHERE",
        "import numpy as np\nimport pandas as pd\n\ndef analyze_data(df):",
        "// Binary search implementation in Java\npublic static int binarySearch(int[] arr, int target) {",
        "# Python decorator for timing functions\nimport time\n\ndef timer(func):",
        "CREATE TABLE products (",
    ]
    
    # Initialize RNG
    rng = jax.random.PRNGKey(42)
    
    print(f"\nProcessing {len(prompts)} prompts in batch...")
    
    # Generate
    results, output = batch_generate(
        model,
        params,
        tokenizer,
        prompts,
        rng,
        max_new_tokens=128,
        temperature=0.4,
        top_p=0.95,
        alg="entropy",
    )
    
    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}:")
        print(prompt)
        print("\nGenerated:")
        print(result)
    
    # Performance statistics
    print(f"\n{'='*60}")
    print("Batch Performance:")
    print(f"  Batch size: {len(prompts)}")
    print(f"  Total tokens generated: {output['sequences'].size}")
    print(f"  Diffusion steps: {len(output['history'])}")
    
    # Memory usage (approximate)
    param_memory = sum(x.nbytes for x in jax.tree_leaves(params)) / 1024**3
    print(f"\nMemory Usage:")
    print(f"  Model parameters: {param_memory:.2f} GB")


if __name__ == "__main__":
    main()