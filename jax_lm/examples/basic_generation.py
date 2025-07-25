#!/usr/bin/env python3
"""Basic generation example for JAX DiffuCoder."""

import jax
import jax.numpy as jnp
from jax_lm import load_model, diffusion_generate
from jax_lm.utils import load_tokenizer


def main():
    # Model path (assumes you've already converted the model)
    model_path = "./models/jax"
    
    print("Loading model...")
    model, params = load_model(model_path, dtype=jnp.bfloat16)
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(f"{model_path}/tokenizer")
    
    # Example prompts
    prompts = [
        "def fibonacci(n):",
        "def quicksort(arr):",
        "class BinarySearchTree:",
        "async function fetchUserData(userId):",
    ]
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="jax")
        input_ids = inputs["input_ids"]
        
        # Generate
        rng, gen_rng = jax.random.split(rng)
        
        output = diffusion_generate(
            model,
            params,
            input_ids,
            gen_rng,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.95,
            alg="entropy",
        )
        
        # Decode
        generated_ids = output["sequences"][0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print("\nGenerated:")
        print(generated_text)
        
        # Show some statistics
        num_steps = len(output["history"])
        print(f"\nStatistics:")
        print(f"  Diffusion steps: {num_steps}")
        print(f"  Tokens generated: {output['sequences'].shape[1] - input_ids.shape[1]}")


if __name__ == "__main__":
    main()