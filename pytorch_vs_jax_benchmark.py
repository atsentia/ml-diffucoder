#!/usr/bin/env python3
"""
Comprehensive PyTorch vs JAX inference benchmark for DiffuCoder.
Tests real model performance and compares frameworks.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add jax_lm to path
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

# Framework imports
try:
    import torch
    import transformers
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch not available")

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    JAX_AVAILABLE = True
    print("✅ JAX available")
except ImportError:
    JAX_AVAILABLE = False
    print("❌ JAX not available")

# Local imports
from jax_lm.models.dream import DreamConfig, DreamForCausalLM


class PyTorchVsJAXBenchmark:
    """Comprehensive benchmark comparing PyTorch and JAX inference performance."""
    
    def __init__(self):
        self.results = {
            "system_info": self._get_system_info(),
            "pytorch_results": {},
            "jax_results": {},
            "comparison": {},
        }
        
        # Real model configuration (will be updated when loading real model)
        self.config = None
        
        # Test prompts
        self.test_prompts = [
            "def fibonacci(n):",
            "class BinaryTree:",
            "def quicksort(arr):",
            "def binary_search(arr, target):",
            "import numpy as np\n\ndef matrix_multiply(A, B):",
        ]
        
        # Generation parameters
        self.gen_params = {
            "max_length": 100,
            "temperature": 0.7,
            "do_sample": True,
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
        }
        
        if PYTORCH_AVAILABLE:
            info["pytorch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        
        if JAX_AVAILABLE:
            info["jax"] = {
                "version": jax.__version__,
                "devices": [str(d) for d in jax.devices()],
                "device_count": jax.device_count(),
            }
        
        return info
    
    def load_real_pytorch_model(self):
        """Load the real PyTorch DiffuCoder model."""
        print("🔧 Loading real PyTorch DiffuCoder model...")
        
        if not PYTORCH_AVAILABLE:
            return None
        
        try:
            import sys
            sys.path.insert(0, './models/diffucoder-7b-complete')
            from modeling_dream import DreamForCausalLM as PyTorchDreamForCausalLM
            from configuration_dream import DreamConfig as PyTorchDreamConfig
            import torch
            
            # Load configuration
            config_path = "./models/diffucoder-7b-complete/config.json"
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            
            config = PyTorchDreamConfig(**config_dict)
            
            # Create model
            model = PyTorchDreamForCausalLM(config)
            
            # Load weights
            from safetensors import safe_open
            import os
            
            model_path = "./models/diffucoder-7b-complete"
            index_file = os.path.join(model_path, "model.safetensors.index.json")
            
            with open(index_file) as f:
                index = json.load(f)
            
            weight_map = index["weight_map"]
            files_to_load = set(weight_map.values())
            
            state_dict = {}
            for file_name in files_to_load:
                file_path = os.path.join(model_path, file_name)
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        if key in weight_map and weight_map[key] == file_name:
                            state_dict[key] = f.get_tensor(key)
            
            # Load weights into model
            model.load_state_dict(state_dict, strict=False)
            
            print(f"✅ Real PyTorch model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load real PyTorch model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def benchmark_pytorch_inference(self, model) -> Dict[str, Any]:
        """Benchmark PyTorch inference."""
        if not PYTORCH_AVAILABLE or model is None:
            return {"error": "PyTorch not available or model not created"}
        
        print("\n🔥 Benchmarking PyTorch inference...")
        
        try:
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            # Load real tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("./models/diffucoder-7b-complete")
            
            def tokenize_text(text):
                tokens = tokenizer.encode(text, return_tensors="pt", max_length=50, truncation=True)
                return tokens.to(device)
            
            results = []
            total_time = 0
            total_tokens = 0
            
            # Warmup
            dummy_input = torch.randint(0, self.config.vocab_size, (1, 10), device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Benchmark each prompt
            for prompt in self.test_prompts:
                print(f"  Testing: {prompt[:30]}...")
                
                # Tokenize
                input_ids = tokenize_text(prompt)
                input_length = input_ids.size(1)
                
                # Time inference
                start_time = time.time()
                
                with torch.no_grad():
                    # Simple generation (just forward pass for demo)
                    outputs = model(input_ids)
                    
                    # Simulate generation by running multiple forward passes
                    current_ids = input_ids
                    for _ in range(self.gen_params["max_length"] - input_length):
                        logits = model(current_ids).logits[:, -1, :]
                        if self.gen_params["do_sample"]:
                            probs = torch.softmax(logits / self.gen_params["temperature"], dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            next_token = logits.argmax(dim=-1, keepdim=True)
                        current_ids = torch.cat([current_ids, next_token], dim=-1)
                        
                        if current_ids.size(1) >= self.gen_params["max_length"]:
                            break
                
                end_time = time.time()
                
                # Calculate metrics
                output_length = current_ids.size(1) - input_length
                inference_time = end_time - start_time
                tokens_per_second = output_length / inference_time if inference_time > 0 else 0
                
                results.append({
                    "prompt": prompt,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                })
                
                total_time += inference_time
                total_tokens += output_length
                
                print(f"    {output_length} tokens in {inference_time:.3f}s ({tokens_per_second:.1f} tok/s)")
            
            # Calculate aggregate metrics
            avg_time = total_time / len(self.test_prompts)
            avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            # Memory usage
            memory_used = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            return {
                "framework": "pytorch",
                "device": device,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "memory_usage_mb": memory_used,
                "total_time": total_time,
                "avg_inference_time": avg_time,
                "avg_tokens_per_second": avg_tokens_per_sec,
                "total_tokens_generated": total_tokens,
                "results": results,
                "success": True,
            }
            
        except Exception as e:
            print(f"❌ PyTorch benchmark failed: {e}")
            return {"framework": "pytorch", "error": str(e), "success": False}
    
    def load_real_jax_weights(self):
        """Load converted JAX weights."""
        print("🔧 Loading converted JAX weights...")
        
        try:
            import pickle
            params_file = "./models/dream-jax/params.pkl"
            
            with open(params_file, "rb") as f:
                params = pickle.load(f)
            
            print("✅ JAX weights loaded successfully")
            return params
            
        except Exception as e:
            print(f"❌ Failed to load JAX weights: {e}")
            return None
    
    def benchmark_jax_inference(self) -> Dict[str, Any]:
        """Benchmark JAX inference."""
        if not JAX_AVAILABLE:
            return {"error": "JAX not available"}
        
        print("\n⚡ Benchmarking JAX inference...")
        
        try:
            # Load real configuration from converted model
            import json
            config_file = "./models/dream-jax/config.json"
            with open(config_file) as f:
                config_dict = json.load(f)
            
            # Create JAX config
            real_config = DreamConfig(
                vocab_size=config_dict["vocab_size"],
                hidden_size=config_dict["hidden_size"],
                intermediate_size=config_dict["intermediate_size"],
                num_hidden_layers=config_dict["num_hidden_layers"],
                num_attention_heads=config_dict["num_attention_heads"],
                num_key_value_heads=config_dict["num_key_value_heads"],
                max_position_embeddings=config_dict["max_position_embeddings"],
                rope_theta=config_dict["rope_theta"],
                rms_norm_eps=config_dict["rms_norm_eps"],
                attention_dropout=config_dict["attention_dropout"],
            )
            
            # Create model
            model = DreamForCausalLM(config=real_config, dtype=real_config.dtype)
            
            # Load real weights
            params = self.load_real_jax_weights()
            if params is None:
                return {"error": "Failed to load JAX weights"}
            
            # JIT compile
            @jax.jit
            def forward_fn(params, input_ids):
                return model.apply(params, input_ids, deterministic=True)
            
            # Load real tokenizer for JAX
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("./models/diffucoder-7b-complete")
            
            def tokenize_text_jax(text):
                tokens = tokenizer.encode(text, max_length=50, truncation=True)
                return jnp.array([tokens])
            
            results = []
            total_time = 0
            total_tokens = 0
            
            # Warmup
            dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
            _ = forward_fn(params, dummy_input)
            
            # Benchmark each prompt
            for prompt in self.test_prompts:
                print(f"  Testing: {prompt[:30]}...")
                
                # Tokenize
                input_ids = tokenize_text_jax(prompt)
                input_length = input_ids.shape[1]
                
                # Time inference
                start_time = time.time()
                
                # Simple generation (forward passes for demo)
                current_ids = input_ids
                for _ in range(self.gen_params["max_length"] - input_length):
                    outputs = forward_fn(params, current_ids)
                    logits = outputs["logits"][:, -1, :]
                    
                    if self.gen_params["do_sample"]:
                        # Sample with temperature
                        probs = jax.nn.softmax(logits / self.gen_params["temperature"])
                        rng, sample_rng = random.split(rng)
                        next_token = random.categorical(sample_rng, logits / self.gen_params["temperature"])
                        next_token = next_token[None, None]
                    else:
                        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
                    
                    current_ids = jnp.concatenate([current_ids, next_token], axis=-1)
                    
                    if current_ids.shape[1] >= self.gen_params["max_length"]:
                        break
                
                end_time = time.time()
                
                # Calculate metrics
                output_length = current_ids.shape[1] - input_length
                inference_time = end_time - start_time
                tokens_per_second = output_length / inference_time if inference_time > 0 else 0
                
                results.append({
                    "prompt": prompt,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                })
                
                total_time += inference_time
                total_tokens += output_length
                
                print(f"    {output_length} tokens in {inference_time:.3f}s ({tokens_per_second:.1f} tok/s)")
            
            # Calculate aggregate metrics
            avg_time = total_time / len(self.test_prompts)
            avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            # Model info
            model_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            
            return {
                "framework": "jax",
                "device": str(jax.devices()[0]),
                "model_parameters": model_params,
                "total_time": total_time,
                "avg_inference_time": avg_time,
                "avg_tokens_per_second": avg_tokens_per_sec,
                "total_tokens_generated": total_tokens,
                "results": results,
                "success": True,
            }
            
        except Exception as e:
            print(f"❌ JAX benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {"framework": "jax", "error": str(e), "success": False}
    
    def run_comparison(self):
        """Run full PyTorch vs JAX comparison."""
        print("🚀 PyTorch vs JAX DiffuCoder Inference Benchmark")
        print("=" * 60)
        
        # Create PyTorch model
        pytorch_model = self.load_real_pytorch_model()
        
        # Run PyTorch benchmark
        print("\n" + "="*60)
        self.results["pytorch_results"] = self.benchmark_pytorch_inference(pytorch_model)
        
        # Run JAX benchmark  
        print("\n" + "="*60)
        self.results["jax_results"] = self.benchmark_jax_inference()
        
        # Compare results
        self._compute_comparison()
        
        return self.results
    
    def _compute_comparison(self):
        """Compute comparison metrics."""
        pytorch_res = self.results["pytorch_results"]
        jax_res = self.results["jax_results"]
        
        if not (pytorch_res.get("success") and jax_res.get("success")):
            self.results["comparison"] = {"error": "One or both benchmarks failed"}
            return
        
        # Speed comparison
        pytorch_speed = pytorch_res.get("avg_tokens_per_second", 0)
        jax_speed = jax_res.get("avg_tokens_per_second", 0)
        
        speed_ratio = jax_speed / pytorch_speed if pytorch_speed > 0 else 0
        
        # Time comparison
        pytorch_time = pytorch_res.get("avg_inference_time", 0)
        jax_time = jax_res.get("avg_inference_time", 0)
        
        time_ratio = pytorch_time / jax_time if jax_time > 0 else 0
        
        # Memory comparison (if available)
        pytorch_memory = pytorch_res.get("memory_usage_mb", 0)
        jax_memory = jax_res.get("memory_usage_mb", 0)
        
        self.results["comparison"] = {
            "speed_comparison": {
                "pytorch_tokens_per_sec": pytorch_speed,
                "jax_tokens_per_sec": jax_speed,
                "jax_speedup": speed_ratio,
                "winner": "JAX" if speed_ratio > 1.0 else "PyTorch",
            },
            "time_comparison": {
                "pytorch_avg_time": pytorch_time,
                "jax_avg_time": jax_time,
                "jax_speedup": time_ratio,
                "winner": "JAX" if time_ratio > 1.0 else "PyTorch",
            },
            "memory_comparison": {
                "pytorch_memory_mb": pytorch_memory,
                "jax_memory_mb": jax_memory,
            },
            "overall_winner": "JAX" if (speed_ratio > 1.0 and time_ratio > 1.0) else "PyTorch"
        }
    
    def print_results(self):
        """Print formatted benchmark results."""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        # System info
        print(f"🖥️  System: {self.results['system_info']['platform']}")
        print(f"🐍 Python: {self.results['system_info']['python_version']}")
        
        if "pytorch" in self.results['system_info']:
            pt_info = self.results['system_info']['pytorch']
            print(f"🔥 PyTorch: {pt_info['version']} (CUDA: {pt_info['cuda_available']})")
        
        if "jax" in self.results['system_info']:
            jax_info = self.results['system_info']['jax']
            print(f"⚡ JAX: {jax_info['version']} ({jax_info['device_count']} devices)")
        
        # Results
        pytorch_res = self.results["pytorch_results"]
        jax_res = self.results["jax_results"]
        
        if pytorch_res.get("success"):
            print(f"\n🔥 PyTorch Results:")
            print(f"   Average tokens/sec: {pytorch_res['avg_tokens_per_second']:.1f}")
            print(f"   Average time/prompt: {pytorch_res['avg_inference_time']:.3f}s")
            print(f"   Memory usage: {pytorch_res.get('memory_usage_mb', 0):.1f} MB")
        else:
            print(f"\n🔥 PyTorch: ❌ {pytorch_res.get('error', 'Failed')}")
        
        if jax_res.get("success"):
            print(f"\n⚡ JAX Results:")
            print(f"   Average tokens/sec: {jax_res['avg_tokens_per_second']:.1f}")
            print(f"   Average time/prompt: {jax_res['avg_inference_time']:.3f}s")
            print(f"   Parameters: {jax_res['model_parameters']:,}")
        else:
            print(f"\n⚡ JAX: ❌ {jax_res.get('error', 'Failed')}")
        
        # Comparison
        if "comparison" in self.results and "error" not in self.results["comparison"]:
            comp = self.results["comparison"]
            print(f"\n📈 Performance Comparison:")
            print(f"   Speed winner: {comp['speed_comparison']['winner']}")
            print(f"   JAX speedup: {comp['speed_comparison']['jax_speedup']:.2f}x")
            print(f"   Time winner: {comp['time_comparison']['winner']}")
            print(f"   Overall winner: {comp['overall_winner']}")
        
        print("="*60)
    
    def save_results(self, filename: str = "pytorch_vs_jax_results.json"):
        """Save results to file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 Results saved to {filename}")


def main():
    """Main benchmark function."""
    try:
        benchmark = PyTorchVsJAXBenchmark()
        results = benchmark.run_comparison()
        benchmark.print_results()
        benchmark.save_results()
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark cancelled by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()