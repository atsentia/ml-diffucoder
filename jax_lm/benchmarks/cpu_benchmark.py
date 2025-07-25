#!/usr/bin/env python3
"""CPU benchmark for JAX DiffuCoder."""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.generate_diffusion import diffusion_generate
from jax_lm.utils import initialize_model


class CPUBenchmark:
    """Benchmark suite for DiffuCoder on CPU."""
    
    def __init__(self, config: DiffuCoderConfig, dtype=jnp.float32):
        self.config = config
        self.dtype = dtype
        self.rng = random.PRNGKey(42)
        
        # Force CPU usage
        jax.config.update('jax_platform_name', 'cpu')
        
        # Initialize model
        print("Initializing model...")
        self.model, self.params = initialize_model(config, self.rng, dtype)
        
        # Count parameters
        self.param_count = sum(x.size for x in jax.tree_leaves(self.params))
        print(f"Model parameters: {self.param_count / 1e6:.2f}M")
    
    def benchmark_forward_pass(
        self,
        batch_sizes: list = [1, 2, 4, 8],
        seq_lengths: list = [128, 256, 512],
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark forward pass performance."""
        results = {"forward_pass": {}}
        
        for batch_size in batch_sizes:
            results["forward_pass"][f"batch_{batch_size}"] = {}
            
            for seq_len in seq_lengths:
                print(f"\nBenchmarking forward pass: batch_size={batch_size}, seq_len={seq_len}")
                
                # Create dummy input
                input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
                attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
                
                # JIT compile
                @jax.jit
                def forward_fn(params, input_ids, attention_mask):
                    return self.model.apply(
                        params,
                        input_ids,
                        attention_mask=attention_mask,
                        deterministic=True,
                    )
                
                # Warmup
                print("Warming up...")
                for _ in range(warmup_runs):
                    outputs = forward_fn(self.params, input_ids, attention_mask)
                    outputs["logits"].block_until_ready()
                
                # Benchmark
                times = []
                for i in range(num_runs):
                    start_time = time.time()
                    outputs = forward_fn(self.params, input_ids, attention_mask)
                    outputs["logits"].block_until_ready()
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")
                
                # Calculate statistics
                times = np.array(times)
                stats = {
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "tokens_per_second": float(batch_size * seq_len / np.mean(times)),
                }
                
                results["forward_pass"][f"batch_{batch_size}"][f"seq_{seq_len}"] = stats
                
                print(f"  Average: {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
                print(f"  Throughput: {stats['tokens_per_second']:.1f} tokens/s")
        
        return results
    
    def benchmark_generation(
        self,
        batch_sizes: list = [1, 2, 4],
        prompt_lengths: list = [32, 64, 128],
        max_new_tokens: int = 128,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """Benchmark generation performance."""
        results = {"generation": {}}
        
        for batch_size in batch_sizes:
            results["generation"][f"batch_{batch_size}"] = {}
            
            for prompt_len in prompt_lengths:
                print(f"\nBenchmarking generation: batch_size={batch_size}, prompt_len={prompt_len}")
                
                # Create dummy input
                input_ids = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
                
                # Generation settings
                num_steps = max_new_tokens // 1  # tokens_per_step = 1
                
                # JIT compile generation
                @jax.jit
                def generate_fn(params, input_ids, rng):
                    return diffusion_generate(
                        self.model,
                        params,
                        input_ids,
                        rng,
                        num_steps=num_steps,
                        tokens_per_step=1,
                        max_new_tokens=max_new_tokens,
                        temperature=0.3,
                        top_p=0.95,
                        alg="entropy",
                    )
                
                # Warmup
                print("Warming up...")
                rng, gen_rng = random.split(self.rng)
                outputs = generate_fn(self.params, input_ids, gen_rng)
                outputs["sequences"].block_until_ready()
                
                # Benchmark
                times = []
                for i in range(num_runs):
                    rng, gen_rng = random.split(rng)
                    
                    start_time = time.time()
                    outputs = generate_fn(self.params, input_ids, gen_rng)
                    outputs["sequences"].block_until_ready()
                    elapsed = time.time() - start_time
                    
                    times.append(elapsed)
                    print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")
                
                # Calculate statistics
                times = np.array(times)
                stats = {
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "tokens_per_second": float(batch_size * max_new_tokens / np.mean(times)),
                    "time_per_token": float(np.mean(times) / (batch_size * max_new_tokens)),
                }
                
                results["generation"][f"batch_{batch_size}"][f"prompt_{prompt_len}"] = stats
                
                print(f"  Average: {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
                print(f"  Throughput: {stats['tokens_per_second']:.1f} tokens/s")
                print(f"  Time per token: {stats['time_per_token']*1000:.1f}ms")
        
        return results
    
    def benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = {
            "memory": {
                "baseline_mb": baseline_memory,
                "model_params_mb": self.param_count * 4 / 1024 / 1024,  # Assuming float32
            }
        }
        
        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            # Forward pass
            input_ids = jnp.ones((batch_size, 256), dtype=jnp.int32)
            outputs = self.model.apply(
                self.params,
                input_ids,
                deterministic=True,
            )
            outputs["logits"].block_until_ready()
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            results["memory"][f"batch_{batch_size}_mb"] = current_memory
            results["memory"][f"batch_{batch_size}_increase_mb"] = current_memory - baseline_memory
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return results."""
        print("=" * 60)
        print("JAX DiffuCoder CPU Benchmark")
        print("=" * 60)
        print(f"Platform: {jax.devices()[0].platform}")
        print(f"Device count: {len(jax.devices())}")
        print(f"Model config: {self.config.num_hidden_layers} layers, "
              f"{self.config.hidden_size} hidden size")
        print("=" * 60)
        
        results = {
            "system_info": {
                "platform": jax.devices()[0].platform,
                "device_count": len(jax.devices()),
                "model_layers": self.config.num_hidden_layers,
                "hidden_size": self.config.hidden_size,
                "param_count": self.param_count,
            }
        }
        
        # Forward pass benchmark
        print("\n1. Forward Pass Benchmark")
        forward_results = self.benchmark_forward_pass(
            batch_sizes=[1, 2, 4],
            seq_lengths=[128, 256],
            num_runs=5,
        )
        results.update(forward_results)
        
        # Generation benchmark
        print("\n2. Generation Benchmark")
        generation_results = self.benchmark_generation(
            batch_sizes=[1, 2],
            prompt_lengths=[32, 64],
            max_new_tokens=64,
            num_runs=3,
        )
        results.update(generation_results)
        
        # Memory benchmark
        print("\n3. Memory Benchmark")
        memory_results = self.benchmark_memory()
        results.update(memory_results)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="CPU benchmark for JAX DiffuCoder")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Model size to benchmark",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_results.json",
        help="Output file for benchmark results",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Data type for model",
    )
    
    args = parser.parse_args()
    
    # Model configurations
    configs = {
        "small": DiffuCoderConfig(
            vocab_size=32000,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
        ),
        "medium": DiffuCoderConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
        ),
        "large": DiffuCoderConfig(
            vocab_size=32000,
            hidden_size=1536,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=24,
        ),
    }
    
    config = configs[args.model_size]
    
    # Get dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Run benchmark
    benchmark = CPUBenchmark(config, dtype)
    results = benchmark.run_all_benchmarks()
    
    # Save results
    output_path = Path(args.output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Forward pass summary
    if "forward_pass" in results:
        print("\nForward Pass Performance:")
        for batch_key, batch_data in results["forward_pass"].items():
            for seq_key, seq_data in batch_data.items():
                print(f"  {batch_key}, {seq_key}: "
                      f"{seq_data['mean_time']:.3f}s "
                      f"({seq_data['tokens_per_second']:.1f} tokens/s)")
    
    # Generation summary
    if "generation" in results:
        print("\nGeneration Performance:")
        for batch_key, batch_data in results["generation"].items():
            for prompt_key, prompt_data in batch_data.items():
                print(f"  {batch_key}, {prompt_key}: "
                      f"{prompt_data['mean_time']:.3f}s "
                      f"({prompt_data['tokens_per_second']:.1f} tokens/s)")
    
    # Memory summary
    if "memory" in results:
        print("\nMemory Usage:")
        mem_data = results["memory"]
        print(f"  Model parameters: {mem_data['model_params_mb']:.1f} MB")
        print(f"  Baseline: {mem_data['baseline_mb']:.1f} MB")
        for key, value in mem_data.items():
            if key.startswith("batch_") and key.endswith("_increase_mb"):
                batch_size = key.split("_")[1]
                print(f"  Batch size {batch_size} increase: {value:.1f} MB")


if __name__ == "__main__":
    main()