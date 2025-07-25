#!/usr/bin/env python3
"""TPU-optimized inference script for DiffuCoder."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random

from jax_lm import (
    load_model,
    diffusion_generate,
)
from jax_lm.utils.tokenizer import (
    load_tokenizer,
    prepare_input_ids,
    decode_sequences,
)
from jax_lm.utils.tpu_utils import (
    setup_tpu,
    get_tpu_mesh,
    shard_params,
)


def main():
    parser = argparse.ArgumentParser(description="Run DiffuCoder inference on TPU")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to JAX model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="def fibonacci(n):",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling threshold",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=256,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=1,
        help="Tokens to unmask per diffusion step",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["entropy", "random"],
        default="entropy",
        help="Algorithm for selecting positions to unmask",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Data type for model",
    )
    
    args = parser.parse_args()
    
    # Setup TPU
    print("Setting up TPU...")
    devices = setup_tpu()
    mesh = get_tpu_mesh(devices)
    
    # Set data type
    if args.dtype == "bfloat16":
        dtype = jnp.bfloat16
    elif args.dtype == "float16":
        dtype = jnp.float16
    else:
        dtype = jnp.float32
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, params = load_model(args.model_path, dtype=dtype)
    
    # Load tokenizer
    tokenizer_path = Path(args.model_path) / "tokenizer"
    if not tokenizer_path.exists():
        # Try parent directory
        tokenizer_path = Path(args.model_path).parent / "tokenizer"
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Shard parameters across TPU devices
    print("Sharding model parameters across TPU devices...")
    with mesh:
        params = shard_params(params, mesh)
    
    # Prepare input
    print(f"Preparing input: '{args.prompt}'")
    
    # Handle batch size
    if args.batch_size > 1:
        prompts = [args.prompt] * args.batch_size
    else:
        prompts = args.prompt
    
    inputs = prepare_input_ids(
        tokenizer,
        prompts,
        max_length=None,
        padding=True,
        truncation=False,
        return_tensors="jax",
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Initialize RNG
    rng = random.PRNGKey(args.seed)
    
    # Compile generation function
    print("Compiling generation function...")
    generate_fn = jax.jit(
        lambda params, input_ids, attention_mask, rng: diffusion_generate(
            model,
            params,
            input_ids,
            rng,
            attention_mask=attention_mask,
            num_steps=args.num_steps,
            tokens_per_step=args.tokens_per_step,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.algorithm,
        ),
        static_argnames=(),
    )
    
    # Warm up
    print("Warming up...")
    with mesh:
        _ = generate_fn(params, input_ids, attention_mask, rng)
    
    # Generate
    print("\nGenerating...")
    start_time = time.time()
    
    with mesh:
        outputs = generate_fn(params, input_ids, attention_mask, rng)
    
    # Wait for completion
    outputs["sequences"].block_until_ready()
    generation_time = time.time() - start_time
    
    # Decode outputs
    generated_sequences = outputs["sequences"]
    
    # Extract only the generated portion
    prompt_length = input_ids.shape[1]
    generated_tokens = generated_sequences[:, prompt_length:]
    
    # Decode
    if args.batch_size > 1:
        generated_texts = decode_sequences(tokenizer, generated_sequences)
        for i, text in enumerate(generated_texts):
            print(f"\n--- Sample {i+1} ---")
            print(text)
    else:
        generated_text = decode_sequences(tokenizer, generated_sequences[0])
        print(f"\n--- Generated Output ---")
        print(generated_text)
    
    # Print statistics
    print(f"\n--- Statistics ---")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens generated: {args.max_new_tokens}")
    print(f"Tokens/second: {args.max_new_tokens / generation_time:.2f}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total tokens/second: {args.max_new_tokens * args.batch_size / generation_time:.2f}")
    
    # Memory statistics
    if devices[0].platform == "tpu":
        memory_stats = devices[0].memory_stats()
        if memory_stats:
            bytes_in_use = memory_stats.get("bytes_in_use", 0) / 1e9
            peak_bytes = memory_stats.get("peak_bytes_in_use", 0) / 1e9
            print(f"\nTPU Memory Usage:")
            print(f"  Current: {bytes_in_use:.2f} GB")
            print(f"  Peak: {peak_bytes:.2f} GB")


if __name__ == "__main__":
    main()