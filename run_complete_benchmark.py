#!/usr/bin/env python3
"""
Comprehensive inference benchmarks for complete DiffuCoder model.
Tests both PyTorch and JAX implementations with real model weights.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Try importing dependencies
try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    from pathlib import Path
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available")

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("⚠️  JAX not available")

class CompleteBenchmarkRunner:
    """Comprehensive benchmark runner for DiffuCoder inference."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.results = {
            "model_path": str(self.model_path),
            "benchmarks": {},
            "system_info": self._get_system_info(),
        }
        
        # Test prompts for code generation
        self.test_prompts = [
            "def fibonacci(n):",
            "class BinaryTree:",
            "def quicksort(arr):",
            "# Function to calculate factorial\ndef factorial(n):",
            "import numpy as np\n\ndef matrix_multiply(A, B):",
            "def is_palindrome(s):",
            "class Calculator:",
            "def merge_sort(arr):",
            "# Binary search implementation\ndef binary_search(arr, target):",
            "def reverse_string(s):",
        ]
        
        # Generation parameters
        self.gen_params = {
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.95,
            "do_sample": True,
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
        }
        
        if PYTORCH_AVAILABLE:
            info["torch_version"] = torch.__version__
            info["torch_cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["torch_cuda_devices"] = torch.cuda.device_count()
                info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
        
        if JAX_AVAILABLE:
            info["jax_version"] = jax.__version__
            info["jax_devices"] = [str(d) for d in jax.devices()]
            info["jax_device_count"] = jax.device_count()
        
        return info
    
    def verify_model_exists(self) -> bool:
        """Verify that the model exists."""
        if not self.model_path.exists():
            print(f"❌ Model not found at {self.model_path}")
            print("Please run: python download_complete_model.py")
            return False
        
        # Check for required files
        required_files = ["config.json"]
        missing_files = [f for f in required_files if not (self.model_path / f).exists()]
        
        if missing_files:
            print(f"⚠️  Missing files: {missing_files}")
            # Try to find model files
            model_files = list(self.model_path.glob("*.bin")) + list(self.model_path.glob("*.safetensors"))
            if not model_files:
                print("❌ No model weight files found")
                return False
        
        print(f"✅ Model verified at {self.model_path}")
        return True
    
    def benchmark_pytorch_inference(self) -> Dict[str, Any]:
        """Benchmark PyTorch inference."""
        if not PYTORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        print("\n🔥 Running PyTorch inference benchmark...")
        
        try:
            # Load model and tokenizer
            print("📥 Loading PyTorch model...")
            start_load = time.time()
            
            # Add the model directory to Python path for custom modules
            model_dir = str(self.model_path.absolute())
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            
            # Try to load model with auto class first, then custom
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"  Auto loading failed: {e}")
                print("  Trying custom Dream model loading...")
                
                # Load using custom model class
                from modeling_dream import DreamForCausalLM
                from configuration_dream import DreamConfig
                
                config = DreamConfig.from_pretrained(str(self.model_path))
                model = DreamForCausalLM.from_pretrained(
                    str(self.model_path),
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                )
            
            load_time = time.time() - start_load
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Get model info
            model_info = {
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
                "memory_usage_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            }
            
            # Warm up
            print("🔥 Warming up...")
            warmup_prompt = "def test():"
            inputs = tokenizer(warmup_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Benchmark inference
            print("⏱️  Running inference benchmarks...")
            inference_times = []
            token_counts = []
            generations = []
            
            for i, prompt in enumerate(self.test_prompts):
                print(f"  Testing prompt {i+1}/{len(self.test_prompts)}: {prompt[:30]}...")
                
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                input_length = inputs["input_ids"].shape[1]
                
                # Generate with timing
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **self.gen_params,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                end_time = time.time()
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = generated_text[len(prompt):].strip()
                
                # Record metrics
                output_length = outputs.shape[1] - input_length
                inference_time = end_time - start_time
                tokens_per_second = output_length / inference_time if inference_time > 0 else 0
                
                inference_times.append(inference_time)
                token_counts.append(output_length)
                generations.append({
                    "prompt": prompt,
                    "completion": completion,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "time_seconds": inference_time,
                    "tokens_per_second": tokens_per_second,
                })
                
                print(f"    Generated {output_length} tokens in {inference_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            
            # Calculate statistics
            avg_inference_time = np.mean(inference_times)
            avg_tokens_per_second = np.mean([g["tokens_per_second"] for g in generations])
            total_tokens_generated = sum(token_counts)
            
            results = {
                "framework": "pytorch",
                "load_time_seconds": load_time,
                "model_info": model_info,
                "inference_metrics": {
                    "avg_inference_time_seconds": avg_inference_time,
                    "avg_tokens_per_second": avg_tokens_per_second,
                    "total_tokens_generated": total_tokens_generated,
                    "num_prompts": len(self.test_prompts),
                },
                "generations": generations,
                "success": True,
            }
            
            print(f"✅ PyTorch benchmark completed!")
            print(f"   Average inference time: {avg_inference_time:.2f}s")
            print(f"   Average tokens/second: {avg_tokens_per_second:.1f}")
            
            return results
            
        except Exception as e:
            print(f"❌ PyTorch benchmark failed: {e}")
            return {
                "framework": "pytorch",
                "error": str(e),
                "success": False,
            }
    
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("🚀 Starting Complete DiffuCoder Inference Benchmark")
        print("=" * 60)
        
        if not self.verify_model_exists():
            return {"error": "Model not found"}
        
        start_time = time.time()
        
        # Run PyTorch benchmark
        print("\n1️⃣  PyTorch Inference Benchmark")
        self.results["benchmarks"]["pytorch_inference"] = self.benchmark_pytorch_inference()
        
        # Calculate total time
        total_time = time.time() - start_time
        self.results["total_benchmark_time_seconds"] = total_time
        
        print(f"\n✅ Complete benchmark finished in {total_time:.2f} seconds")
        
        return self.results
    
    def save_results(self, output_file: str = "complete_benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = Path(output_file)
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📁 Results saved to {output_path}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        # PyTorch results
        if "pytorch_inference" in self.results["benchmarks"]:
            pt_results = self.results["benchmarks"]["pytorch_inference"]
            if pt_results.get("success"):
                metrics = pt_results["inference_metrics"]
                print(f"🔥 PyTorch:")
                print(f"   Load time: {pt_results['load_time_seconds']:.2f}s")
                print(f"   Avg inference: {metrics['avg_inference_time_seconds']:.2f}s")
                print(f"   Avg tokens/sec: {metrics['avg_tokens_per_second']:.1f}")
                print(f"   Total tokens: {metrics['total_tokens_generated']}")
            else:
                print(f"🔥 PyTorch: ❌ {pt_results.get('error', 'Failed')}")
        
        print("=" * 60)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete DiffuCoder inference benchmark")
    parser.add_argument(
        "--model-path",
        default="./models/diffucoder-7b-complete",
        help="Path to DiffuCoder model"
    )
    parser.add_argument(
        "--output",
        default="complete_benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    try:
        # Run benchmark
        runner = CompleteBenchmarkRunner(args.model_path)
        results = runner.run_complete_benchmark()
        runner.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark cancelled by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()