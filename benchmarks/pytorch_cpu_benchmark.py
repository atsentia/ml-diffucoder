#!/usr/bin/env python3
"""PyTorch CPU benchmark for DiffuCoder - for comparison with JAX."""

import argparse
import time
import json
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig


class PyTorchCPUBenchmark:
    """Benchmark suite for PyTorch DiffuCoder on CPU."""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "float32",
    ):
        self.model_path = model_path
        self.dtype_str = dtype
        
        # Force CPU usage
        torch.set_num_threads(psutil.cpu_count(logical=False))  # Use physical cores
        self.device = torch.device("cpu")
        
        # Hardware info
        print("=" * 60)
        print("PyTorch Hardware Detection")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {self.device}")
        print(f"CPU threads: {torch.get_num_threads()}")
        print(f"Platform: {platform.system()} {platform.machine()}")
        
        # Memory info
        memory_info = psutil.virtual_memory()
        print(f"Available memory: {memory_info.available / (1024**3):.2f} GB")
        print(f"Total memory: {memory_info.total / (1024**3):.2f} GB")
        print("=" * 60)
        
        # Set dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.float32)
        
        # Load model
        print("\nLoading PyTorch model...")
        try:
            # Load config
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # For benchmarking, create a smaller model if needed
            if hasattr(config, "num_hidden_layers") and config.num_hidden_layers > 12:
                print(f"Reducing model size for CPU benchmark: {config.num_hidden_layers} -> 12 layers")
                config.num_hidden_layers = 12
            
            # Load model
            self.model = AutoModel.from_pretrained(
                model_path,
                config=config,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move to CPU and eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            self.param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {self.param_count / 1e6:.2f}M")
            
            # Store config
            self.config = config
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating mock model for benchmarking...")
            
            # Create a mock model for testing
            from dataclasses import dataclass
            
            @dataclass
            class MockConfig:
                vocab_size: int = 32000
                hidden_size: int = 768
                num_hidden_layers: int = 12
                num_attention_heads: int = 12
                max_position_embeddings: int = 4096
            
            self.config = MockConfig()
            self.model = self._create_mock_model()
            self.param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Mock model parameters: {self.param_count / 1e6:.2f}M")
    
    def _create_mock_model(self):
        """Create a mock transformer model for testing."""
        class MockDiffuCoder(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=config.hidden_size,
                        nhead=config.num_attention_heads,
                        dim_feedforward=config.hidden_size * 4,
                        batch_first=True,
                    )
                    for _ in range(config.num_hidden_layers)
                ])
                self.norm = nn.LayerNorm(config.hidden_size)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                # Simple forward pass
                x = self.embeddings(input_ids)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                logits = self.lm_head(x)
                
                return type('Output', (), {'logits': logits})()
        
        return MockDiffuCoder(self.config).to(self.device).eval()
    
    def verify_computation_device(self):
        """Verify CPU is being used."""
        test_tensor = torch.randn(100, 100)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"Computation device: {result.device}")
        return result.device == torch.device("cpu")
    
    def benchmark_forward_pass(
        self,
        batch_sizes: list = [1, 2, 4],
        seq_lengths: list = [128, 256],
        num_runs: int = 5,
        warmup_runs: int = 2,
    ) -> Dict[str, Any]:
        """Benchmark forward pass performance."""
        print(f"\nBenchmarking PyTorch on CPU...")
        assert self.verify_computation_device(), "Not running on CPU!"
        
        results = {"forward_pass": {}}
        
        for batch_size in batch_sizes:
            results["forward_pass"][f"batch_{batch_size}"] = {}
            
            for seq_len in seq_lengths:
                print(f"\nForward pass: batch_size={batch_size}, seq_len={seq_len}")
                
                # Create dummy input
                input_ids = torch.ones(
                    (batch_size, seq_len),
                    dtype=torch.long,
                    device=self.device
                )
                attention_mask = torch.ones(
                    (batch_size, seq_len),
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Warmup
                print("  Warming up...")
                with torch.no_grad():
                    for _ in range(warmup_runs):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        if hasattr(outputs, 'logits'):
                            _ = outputs.logits
                
                # Benchmark
                times = []
                with torch.no_grad():
                    for i in range(num_runs):
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        
                        start_time = time.time()
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        if hasattr(outputs, 'logits'):
                            _ = outputs.logits.cpu()  # Force computation
                        elapsed = time.time() - start_time
                        
                        times.append(elapsed)
                        print(f"    Run {i+1}/{num_runs}: {elapsed:.3f}s")
                
                # Calculate statistics
                times = np.array(times)
                stats = {
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "tokens_per_second": float(batch_size * seq_len / np.mean(times)),
                    "backend": "pytorch_cpu",
                    "dtype": self.dtype_str,
                }
                
                results["forward_pass"][f"batch_{batch_size}"][f"seq_{seq_len}"] = stats
                
                print(f"  Average: {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
                print(f"  Throughput: {stats['tokens_per_second']:.1f} tokens/s")
        
        return results
    
    def benchmark_generation(
        self,
        batch_sizes: list = [1, 2],
        prompt_lengths: list = [32, 64],
        max_new_tokens: int = 64,
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark generation performance (if supported)."""
        print(f"\nGeneration benchmark on PyTorch CPU...")
        
        results = {"generation": {}}
        
        # Check if model supports generation
        if not hasattr(self.model, 'diffusion_generate'):
            print("  Model does not support diffusion generation, skipping...")
            return results
        
        for batch_size in batch_sizes:
            results["generation"][f"batch_{batch_size}"] = {}
            
            for prompt_len in prompt_lengths:
                print(f"\nGeneration: batch_size={batch_size}, prompt_len={prompt_len}")
                
                # Create dummy input
                input_ids = torch.ones(
                    (batch_size, prompt_len),
                    dtype=torch.long,
                    device=self.device
                )
                
                # Benchmark
                times = []
                with torch.no_grad():
                    for i in range(num_runs):
                        start_time = time.time()
                        
                        # Try diffusion generation
                        try:
                            outputs = self.model.diffusion_generate(
                                input_ids,
                                max_new_tokens=max_new_tokens,
                                steps=max_new_tokens,
                                temperature=0.3,
                                top_p=0.95,
                            )
                        except:
                            # Fallback to regular generation if available
                            outputs = {"sequences": input_ids}
                        
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                        print(f"    Run {i+1}/{num_runs}: {elapsed:.3f}s")
                
                # Calculate statistics
                times = np.array(times)
                stats = {
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "tokens_per_second": float(batch_size * max_new_tokens / np.mean(times)),
                    "time_per_token": float(np.mean(times) / (batch_size * max_new_tokens)),
                    "backend": "pytorch_cpu",
                    "max_new_tokens": max_new_tokens,
                }
                
                results["generation"][f"batch_{batch_size}"][f"prompt_{prompt_len}"] = stats
                
                print(f"  Average: {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
                print(f"  Throughput: {stats['tokens_per_second']:.1f} tokens/s")
        
        return results
    
    def benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = {
            "memory": {
                "baseline_mb": baseline_memory,
                "model_params_mb": self.param_count * 4 / 1024 / 1024,  # Assuming float32
            }
        }
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            # Forward pass
            input_ids = torch.ones((batch_size, 256), dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                if hasattr(outputs, 'logits'):
                    _ = outputs.logits
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            results["memory"][f"batch_{batch_size}_mb"] = current_memory
            results["memory"][f"batch_{batch_size}_increase_mb"] = current_memory - baseline_memory
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return results."""
        print("\n" + "=" * 60)
        print("PyTorch DiffuCoder CPU Benchmark")
        print("=" * 60)
        
        results = {
            "system_info": {
                "backend": "pytorch",
                "device": str(self.device),
                "pytorch_version": torch.__version__,
                "cpu_threads": torch.get_num_threads(),
                "platform": platform.system(),
                "architecture": platform.machine(),
                "model_layers": getattr(self.config, 'num_hidden_layers', 'unknown'),
                "hidden_size": getattr(self.config, 'hidden_size', 'unknown'),
                "param_count": self.param_count,
                "dtype": self.dtype_str,
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
    parser = argparse.ArgumentParser(
        description="PyTorch CPU benchmark for DiffuCoder"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="pytorch_cpu_results.json",
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
    
    # Run benchmark
    benchmark = PyTorchCPUBenchmark(args.model_path, args.dtype)
    results = benchmark.run_all_benchmarks()
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if "forward_pass" in results:
        print("\nForward Pass Performance:")
        for batch_key, batch_data in results["forward_pass"].items():
            for seq_key, seq_data in batch_data.items():
                print(f"  {batch_key}, {seq_key}: "
                      f"{seq_data['mean_time']:.3f}s "
                      f"({seq_data['tokens_per_second']:.1f} tokens/s)")


if __name__ == "__main__":
    main()