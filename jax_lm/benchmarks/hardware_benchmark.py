#!/usr/bin/env python3
"""Multi-hardware benchmark for JAX DiffuCoder with automatic device selection."""

import argparse
import time
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.generate_diffusion import diffusion_generate
from jax_lm.utils import initialize_model


class HardwareDetector:
    """Detect and select appropriate hardware for JAX."""
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available JAX backends."""
        available = []
        
        # Check CPU (always available)
        available.append("cpu")
        
        # Check CUDA/GPU
        try:
            # Try to get GPU info
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                available.append("gpu")
        except:
            pass
        
        # Check TPU
        try:
            # TPU detection via environment or JAX
            if "TPU_NAME" in os.environ:
                available.append("tpu")
            else:
                # Try to initialize TPU backend
                jax.config.update('jax_platform_name', 'tpu')
                if len(jax.devices()) > 0 and jax.devices()[0].platform == "tpu":
                    available.append("tpu")
        except:
            pass
        
        return available
    
    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Get detailed hardware information."""
        info = {
            "system": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            "jax": {
                "version": jax.__version__,
                "backends": HardwareDetector.get_available_backends(),
            }
        }
        
        # CPU info
        try:
            import psutil
            info["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            }
        except:
            pass
        
        # GPU info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpus.append({
                            "name": parts[0],
                            "memory_mb": parts[1],
                            "driver_version": parts[2],
                        })
                info["gpu"] = gpus
        except:
            pass
        
        return info
    
    @staticmethod
    def select_backend(preferred: Optional[str] = None) -> str:
        """Select the best available backend."""
        available = HardwareDetector.get_available_backends()
        
        if preferred and preferred in available:
            return preferred
        
        # Priority order: TPU > GPU > CPU
        for backend in ["tpu", "gpu", "cpu"]:
            if backend in available:
                return backend
        
        return "cpu"  # Fallback


class MultiHardwareBenchmark:
    """Benchmark suite for DiffuCoder on multiple hardware types."""
    
    def __init__(
        self,
        config: DiffuCoderConfig,
        backend: str = "auto",
        dtype: str = "float32",
    ):
        self.config = config
        self.dtype_str = dtype
        self.rng = random.PRNGKey(42)
        
        # Hardware detection and selection
        print("=" * 60)
        print("Hardware Detection")
        print("=" * 60)
        
        hardware_info = HardwareDetector.get_hardware_info()
        print(f"System: {hardware_info['system']['platform']} {hardware_info['system']['architecture']}")
        print(f"JAX version: {hardware_info['jax']['version']}")
        print(f"Available backends: {hardware_info['jax']['backends']}")
        
        # Select backend
        if backend == "auto":
            self.backend = HardwareDetector.select_backend()
            print(f"Auto-selected backend: {self.backend}")
        else:
            self.backend = backend
            if self.backend not in hardware_info['jax']['backends']:
                print(f"WARNING: Requested backend '{self.backend}' not available")
                self.backend = HardwareDetector.select_backend()
                print(f"Falling back to: {self.backend}")
        
        # Configure JAX for selected backend
        if self.backend == "cpu":
            jax.config.update('jax_platform_name', 'cpu')
        elif self.backend == "gpu":
            jax.config.update('jax_platform_name', 'cuda')
        elif self.backend == "tpu":
            jax.config.update('jax_platform_name', 'tpu')
        
        # Verify device configuration
        devices = jax.devices()
        print(f"\nActive JAX devices:")
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device.device_kind} (platform: {device.platform})")
        
        if len(devices) == 0:
            raise RuntimeError("No JAX devices found!")
        
        actual_platform = devices[0].platform
        print(f"\nActual platform in use: {actual_platform.upper()}")
        
        # Verify we're using the intended backend
        expected_platform_map = {
            "cpu": "cpu",
            "gpu": "gpu",
            "tpu": "tpu"
        }
        
        if actual_platform != expected_platform_map.get(self.backend, self.backend):
            print(f"WARNING: Expected {self.backend}, but JAX is using {actual_platform}")
        
        print("=" * 60)
        
        # Set dtype
        dtype_map = {
            "float32": jnp.float32,
            "float16": jnp.float16,
            "bfloat16": jnp.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, jnp.float32)
        
        # Initialize model
        print("\nInitializing model...")
        self.model, self.params = initialize_model(config, self.rng, self.dtype)
        
        # Count parameters
        self.param_count = sum(x.size for x in jax.tree_leaves(self.params))
        print(f"Model parameters: {self.param_count / 1e6:.2f}M")
        
        # Store hardware info
        self.hardware_info = hardware_info
        self.hardware_info["active_backend"] = self.backend
        self.hardware_info["active_platform"] = actual_platform
        self.hardware_info["device_count"] = len(devices)
    
    def verify_computation_device(self, operation_name: str = ""):
        """Verify which device is actually being used for computation."""
        # Create a test computation
        test_array = jnp.ones((100, 100))
        result = jnp.dot(test_array, test_array)
        
        # Check device placement
        if hasattr(result, "device"):
            device = result.device()
            print(f"{operation_name} running on: {device}")
        
        return result
    
    def benchmark_forward_pass(
        self,
        batch_sizes: list = [1, 2, 4, 8],
        seq_lengths: list = [128, 256, 512],
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark forward pass performance."""
        print(f"\nBenchmarking on {self.backend.upper()}...")
        self.verify_computation_device("Forward pass")
        
        results = {"forward_pass": {}}
        
        for batch_size in batch_sizes:
            # Skip large batches on CPU
            if self.backend == "cpu" and batch_size > 4:
                continue
                
            results["forward_pass"][f"batch_{batch_size}"] = {}
            
            for seq_len in seq_lengths:
                # Skip long sequences on CPU
                if self.backend == "cpu" and seq_len > 256:
                    continue
                
                print(f"\nForward pass: batch_size={batch_size}, seq_len={seq_len}")
                
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
                print("  Warming up...")
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
                
                # Calculate statistics
                times = np.array(times)
                stats = {
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "tokens_per_second": float(batch_size * seq_len / np.mean(times)),
                    "backend": self.backend,
                    "platform": self.hardware_info["active_platform"],
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
        print(f"\nGeneration benchmark on {self.backend.upper()}...")
        self.verify_computation_device("Generation")
        
        results = {"generation": {}}
        
        for batch_size in batch_sizes:
            # Adjust for hardware limitations
            if self.backend == "cpu" and batch_size > 2:
                continue
                
            results["generation"][f"batch_{batch_size}"] = {}
            
            for prompt_len in prompt_lengths:
                print(f"\nGeneration: batch_size={batch_size}, prompt_len={prompt_len}")
                
                # Create dummy input
                input_ids = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
                
                # Adjust generation length for CPU
                actual_max_tokens = max_new_tokens if self.backend != "cpu" else min(64, max_new_tokens)
                num_steps = actual_max_tokens // 1  # tokens_per_step = 1
                
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
                        max_new_tokens=actual_max_tokens,
                        temperature=0.3,
                        top_p=0.95,
                        alg="entropy",
                    )
                
                # Warmup
                print("  Warming up...")
                rng, gen_rng = random.split(self.rng)
                outputs = generate_fn(self.params, input_ids, gen_rng)
                outputs["sequences"].block_until_ready()
                
                # Benchmark
                times = []
                actual_runs = min(num_runs, 3) if self.backend == "cpu" else num_runs
                
                for i in range(actual_runs):
                    rng, gen_rng = random.split(rng)
                    
                    start_time = time.time()
                    outputs = generate_fn(self.params, input_ids, gen_rng)
                    outputs["sequences"].block_until_ready()
                    elapsed = time.time() - start_time
                    
                    times.append(elapsed)
                    print(f"  Run {i+1}/{actual_runs}: {elapsed:.3f}s")
                
                # Calculate statistics
                times = np.array(times)
                stats = {
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "tokens_per_second": float(batch_size * actual_max_tokens / np.mean(times)),
                    "time_per_token": float(np.mean(times) / (batch_size * actual_max_tokens)),
                    "backend": self.backend,
                    "platform": self.hardware_info["active_platform"],
                    "max_new_tokens": actual_max_tokens,
                }
                
                results["generation"][f"batch_{batch_size}"][f"prompt_{prompt_len}"] = stats
                
                print(f"  Average: {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
                print(f"  Throughput: {stats['tokens_per_second']:.1f} tokens/s")
                print(f"  Time per token: {stats['time_per_token']*1000:.1f}ms")
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return results."""
        print("\n" + "=" * 60)
        print(f"JAX DiffuCoder Benchmark on {self.backend.upper()}")
        print("=" * 60)
        
        results = {
            "hardware_info": self.hardware_info,
            "system_info": {
                "backend": self.backend,
                "platform": self.hardware_info["active_platform"],
                "device_count": self.hardware_info["device_count"],
                "model_layers": self.config.num_hidden_layers,
                "hidden_size": self.config.hidden_size,
                "param_count": self.param_count,
                "dtype": self.dtype_str,
            }
        }
        
        # Adjust benchmark parameters based on backend
        if self.backend == "cpu":
            batch_sizes = [1, 2]
            seq_lengths = [128, 256]
            num_runs = 3
        elif self.backend == "gpu":
            batch_sizes = [1, 2, 4, 8]
            seq_lengths = [128, 256, 512]
            num_runs = 10
        else:  # TPU
            batch_sizes = [1, 2, 4, 8, 16]
            seq_lengths = [128, 256, 512, 1024]
            num_runs = 10
        
        # Forward pass benchmark
        print("\n1. Forward Pass Benchmark")
        forward_results = self.benchmark_forward_pass(
            batch_sizes=batch_sizes,
            seq_lengths=seq_lengths,
            num_runs=num_runs,
        )
        results.update(forward_results)
        
        # Generation benchmark
        print("\n2. Generation Benchmark")
        generation_results = self.benchmark_generation(
            batch_sizes=batch_sizes[:3],  # Limit batch sizes for generation
            prompt_lengths=[32, 64],
            max_new_tokens=128 if self.backend != "cpu" else 64,
            num_runs=5 if self.backend != "cpu" else 3,
        )
        results.update(generation_results)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-hardware benchmark for JAX DiffuCoder"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "cpu", "gpu", "tpu"],
        default="auto",
        help="Hardware backend to use (auto will select best available)",
    )
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
    
    # Run benchmark
    try:
        benchmark = MultiHardwareBenchmark(config, args.backend, args.dtype)
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
        print(f"Backend: {results['system_info']['backend']}")
        print(f"Platform: {results['system_info']['platform']}")
        print(f"Device count: {results['system_info']['device_count']}")
        
        # Performance summary
        if "forward_pass" in results:
            print("\nForward Pass Performance:")
            for batch_key, batch_data in results["forward_pass"].items():
                for seq_key, seq_data in batch_data.items():
                    print(f"  {batch_key}, {seq_key}: "
                          f"{seq_data['mean_time']:.3f}s "
                          f"({seq_data['tokens_per_second']:.1f} tokens/s)")
    
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os  # Add this import at the module level
    main()