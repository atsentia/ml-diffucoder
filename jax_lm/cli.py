#!/usr/bin/env python3
"""Command-line interface for JAX-DiffuCoder."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

from jax_lm import (
    __version__,
    load_model,
    load_tokenizer,
    generate,
    batch_generate,
    save_for_huggingface
)
from jax_lm.utils.model_utils import count_parameters, parameter_summary


def cmd_generate(args):
    """Run text generation."""
    print(f"Loading model from {args.model}...")
    
    # Set dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16
    }
    dtype = dtype_map.get(args.dtype, jnp.float32)
    
    # Load model
    model, params = load_model(args.model, dtype=dtype)
    tokenizer = load_tokenizer(args.model)
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()
    else:
        print("Enter prompt (Ctrl+D to finish):")
        prompt = sys.stdin.read().strip()
    
    # Generate
    print(f"\nGenerating with temperature={args.temperature}, max_tokens={args.max_tokens}...")
    output = generate(
        model, params, prompt, tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed
    )
    
    # Output
    if args.output:
        Path(args.output).write_text(output)
        print(f"Output saved to {args.output}")
    else:
        print("\n" + "="*60)
        print(output)
        print("="*60)


def cmd_batch_generate(args):
    """Run batch generation."""
    print(f"Loading model from {args.model}...")
    
    # Load model
    dtype = getattr(jnp, args.dtype)
    model, params = load_model(args.model, dtype=dtype)
    tokenizer = load_tokenizer(args.model)
    
    # Load prompts
    if args.prompts_file.endswith('.json'):
        prompts = json.loads(Path(args.prompts_file).read_text())
    else:
        prompts = Path(args.prompts_file).read_text().strip().split('\n')
    
    print(f"Generating {len(prompts)} outputs...")
    
    # Generate
    outputs = batch_generate(
        model, params, prompts, tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed
    )
    
    # Save outputs
    if args.output:
        if args.output.endswith('.json'):
            result = [{"prompt": p, "output": o} for p, o in zip(prompts, outputs)]
            Path(args.output).write_text(json.dumps(result, indent=2))
        else:
            Path(args.output).write_text('\n'.join(outputs))
        print(f"Outputs saved to {args.output}")
    else:
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n--- Output {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")


def cmd_info(args):
    """Show model information."""
    print(f"JAX-DiffuCoder v{__version__}")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    if args.model:
        print(f"\nLoading model info from {args.model}...")
        
        # Try to load just config
        try:
            from jax_lm.utils.pure_jax_loader import PureJAXModelLoader
            loader = PureJAXModelLoader()
            
            # Download just config
            local_dir = loader.download_model_files(
                args.model, files=["config.json"]
            )
            config = loader.load_config(local_dir / "config.json")
            
            print(f"\nModel Configuration:")
            print(f"  Hidden size: {config.hidden_size}")
            print(f"  Layers: {config.num_hidden_layers}")
            print(f"  Attention heads: {config.num_attention_heads}")
            print(f"  Vocabulary size: {config.vocab_size}")
            print(f"  Max position: {config.max_position_embeddings}")
            
            # Estimate parameter count
            estimated_params = (
                config.vocab_size * config.hidden_size * 2 +  # embeddings
                config.num_hidden_layers * (
                    4 * config.hidden_size * config.hidden_size +  # attention
                    3 * config.hidden_size * config.intermediate_size  # mlp
                )
            )
            print(f"  Estimated parameters: {estimated_params:,}")
            
        except Exception as e:
            print(f"Could not load model info: {e}")


def cmd_convert(args):
    """Convert model to HuggingFace format."""
    from jax_lm.utils.model_utils import load_model
    
    print(f"Loading model from {args.input}...")
    
    # Load model
    dtype = getattr(jnp, args.dtype)
    model, params = load_model(args.input, dtype=dtype)
    
    # Get parameter count
    param_count = count_parameters(params)
    print(f"Model has {param_count:,} parameters")
    
    if args.summary:
        summary = parameter_summary(params)
        print("\nParameter Summary:")
        for layer_type, count in summary.items():
            print(f"  {layer_type}: {count:,}")
    
    # Save for HuggingFace
    print(f"\nConverting to HuggingFace format...")
    save_for_huggingface(
        model,
        params,
        Path(args.output),
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
        max_shard_size=int(args.shard_size[:-2]) * 1024**3  # Convert GB to bytes
    )
    
    print(f"✅ Model saved to {args.output}")


def cmd_benchmark(args):
    """Run performance benchmark."""
    from jax_lm.benchmarks.hardware_benchmark import run_benchmark
    
    print(f"Running benchmark on {args.backend}...")
    
    results = run_benchmark(
        model_path=args.model,
        backend=args.backend,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    # Display results
    print("\nBenchmark Results:")
    print("-" * 60)
    for config, metrics in results.items():
        print(f"\n{config}:")
        print(f"  Throughput: {metrics['throughput']:.2f} tokens/sec")
        print(f"  Latency: {metrics['latency']:.2f} ms/token")
        print(f"  Memory: {metrics['memory_mb']:.1f} MB")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="JAX-DiffuCoder: High-performance code generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"jax-diffucoder {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--model", "-m", default="atsentia/DiffuCoder-7B-JAX")
    gen_parser.add_argument("--prompt", "-p", help="Input prompt")
    gen_parser.add_argument("--prompt-file", "-f", help="Read prompt from file")
    gen_parser.add_argument("--output", "-o", help="Output file")
    gen_parser.add_argument("--max-tokens", type=int, default=256)
    gen_parser.add_argument("--temperature", type=float, default=0.7)
    gen_parser.add_argument("--top-p", type=float, default=0.9)
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--dtype", default="float32",
                          choices=["float32", "float16", "bfloat16"])
    
    # Batch generate command
    batch_parser = subparsers.add_parser("batch", help="Batch generation")
    batch_parser.add_argument("prompts_file", help="File with prompts (one per line)")
    batch_parser.add_argument("--model", "-m", default="atsentia/DiffuCoder-7B-JAX")
    batch_parser.add_argument("--output", "-o", help="Output file")
    batch_parser.add_argument("--max-tokens", type=int, default=256)
    batch_parser.add_argument("--temperature", type=float, default=0.7)
    batch_parser.add_argument("--seed", type=int, default=42)
    batch_parser.add_argument("--dtype", default="float32")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show model/system info")
    info_parser.add_argument("--model", "-m", help="Model to inspect")
    
    # Convert command
    conv_parser = subparsers.add_parser("convert", help="Convert model format")
    conv_parser.add_argument("input", help="Input model path")
    conv_parser.add_argument("output", help="Output directory")
    conv_parser.add_argument("--tokenizer", help="Tokenizer path")
    conv_parser.add_argument("--shard-size", default="5GB")
    conv_parser.add_argument("--dtype", default="float32")
    conv_parser.add_argument("--summary", action="store_true",
                           help="Show parameter summary")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument("--model", "-m", default="atsentia/DiffuCoder-7B-JAX")
    bench_parser.add_argument("--backend", default="auto",
                            choices=["auto", "cpu", "gpu", "tpu"])
    bench_parser.add_argument("--batch-sizes", nargs="+", type=int,
                            default=[1, 2, 4, 8])
    bench_parser.add_argument("--sequence-lengths", nargs="+", type=int,
                            default=[128, 256, 512])
    bench_parser.add_argument("--iterations", type=int, default=10)
    bench_parser.add_argument("--warmup", type=int, default=3)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    commands = {
        "generate": cmd_generate,
        "batch": cmd_batch_generate,
        "info": cmd_info,
        "convert": cmd_convert,
        "benchmark": cmd_benchmark
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()