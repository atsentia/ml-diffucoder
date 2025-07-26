#!/usr/bin/env python3
"""
Comprehensive PyTorch vs JAX inference benchmark for DiffuCoder with real weights.
Fixed version that properly loads both models.
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
    from transformers import AutoTokenizer
    PYTORCH_AVAILABLE = True
    print(f"✅ PyTorch {torch.__version__} available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch not available")

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    JAX_AVAILABLE = True
    print(f"✅ JAX {jax.__version__} available")
except ImportError:
    JAX_AVAILABLE = False
    print("❌ JAX not available")

# Local imports
if JAX_AVAILABLE:
    from jax_lm.models.dream import DreamConfig, DreamForCausalLM


class RealModelBenchmark:
    """Benchmark comparing real PyTorch and JAX DiffuCoder models."""
    
    def __init__(self):
        self.results = {
            "system_info": self._get_system_info(),
            "pytorch_results": {},
            "jax_results": {},
            "comparison": {},
        }
        
        # Test prompts for code generation
        self.test_prompts = [
            "def fibonacci(n):",
            "class BinaryTree:",
            "def quicksort(arr):",
            "def binary_search(arr, target):",
            "import numpy as np\n\ndef matrix_multiply(A, B):",
        ]
        
        # Generation parameters
        self.gen_params = {
            "max_new_tokens": 50,  # Reduced for faster benchmarking
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
                "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        
        if JAX_AVAILABLE:
            info["jax"] = {
                "version": jax.__version__,
                "devices": [str(d) for d in jax.devices()],
                "device_count": jax.device_count(),
            }
        
        return info
    
    def load_pytorch_model(self):
        """Load the real PyTorch DiffuCoder model."""
        print("\n🔧 Loading real PyTorch DiffuCoder model...")
        
        if not PYTORCH_AVAILABLE:
            return None, None, None
        
        try:
            # Use the same approach as the Colab notebook
            from transformers import AutoModel, AutoTokenizer
            
            model_path = "./models/diffucoder-7b-complete"
            
            # Load tokenizer
            print("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            print(f"    Tokenizer vocab size: {tokenizer.vocab_size}")
            
            # Determine device and dtype
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.bfloat16
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32  # MPS doesn't support bfloat16
            else:
                device = "cpu"
                dtype = torch.float32
            
            print(f"    Using device: {device}, dtype: {dtype}")
            
            # Load model
            print("  Loading model weights...")
            start_time = time.time()
            
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            
            # Move to device
            model = model.to(device).eval()
            
            load_time = time.time() - start_time
            
            # Get model info
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✅ Model loaded in {load_time:.2f}s")
            print(f"     Parameters: {num_params/1e9:.1f}B")
            print(f"     Device: {device}")
            print(f"     Dtype: {dtype}")
            
            # Get config from model
            config = model.config
            
            return model, tokenizer, config
            
        except Exception as e:
            print(f"  ❌ Failed to load PyTorch model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def benchmark_pytorch(self, model, tokenizer, config) -> Dict[str, Any]:
        """Benchmark PyTorch inference."""
        if not PYTORCH_AVAILABLE or model is None:
            return {"error": "PyTorch not available or model not loaded"}
        
        print("\n🔥 Benchmarking PyTorch inference...")
        
        try:
            model.eval()
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            
            results = []
            total_time = 0
            total_tokens = 0
            
            # Warmup
            print("  Warming up...")
            dummy_input = torch.randint(0, config.vocab_size, (1, 10), device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Benchmark each prompt
            for i, prompt in enumerate(self.test_prompts):
                print(f"  [{i+1}/{len(self.test_prompts)}] Testing: {prompt[:30]}...")
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_length = inputs["input_ids"].shape[1]
                
                # Time generation
                torch.cuda.synchronize() if device.type == "cuda" else None
                start_time = time.time()
                
                with torch.no_grad():
                    # Check if model has diffusion_generate method (as shown in Colab)
                    if hasattr(model, 'diffusion_generate'):
                        try:
                            # Use diffusion_generate like in the Colab notebook
                            output = model.diffusion_generate(
                                inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=self.gen_params["max_new_tokens"],
                                steps=self.gen_params["max_new_tokens"],  # TOKEN_PER_STEP = 1
                                temperature=self.gen_params["temperature"],
                                top_p=0.95,
                                alg="entropy",
                                alg_temp=0.,
                                output_history=False,
                                return_dict_in_generate=True,
                            )
                            outputs = output.sequences
                        except Exception as e:
                            print(f"    Diffusion generate failed: {e}, using manual generation")
                            outputs = self._manual_generate_pytorch(
                                model, inputs["input_ids"], 
                                max_new_tokens=self.gen_params["max_new_tokens"],
                                temperature=self.gen_params["temperature"],
                                device=device
                            )
                    else:
                        # Use standard generate method
                        try:
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=self.gen_params["max_new_tokens"],
                                temperature=self.gen_params["temperature"],
                                do_sample=self.gen_params["do_sample"],
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        except Exception as e:
                            print(f"    Generate failed: {e}, using manual generation")
                            outputs = self._manual_generate_pytorch(
                                model, inputs["input_ids"], 
                                max_new_tokens=self.gen_params["max_new_tokens"],
                                temperature=self.gen_params["temperature"],
                                device=device
                            )
                
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()
                
                # Calculate metrics
                output_length = outputs.shape[1] - input_length
                inference_time = end_time - start_time
                tokens_per_second = output_length / inference_time if inference_time > 0 else 0
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = generated_text[len(prompt):]
                
                results.append({
                    "prompt": prompt,
                    "completion": completion[:100] + "..." if len(completion) > 100 else completion,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                })
                
                total_time += inference_time
                total_tokens += output_length
                
                print(f"    Generated {output_length} tokens in {inference_time:.3f}s ({tokens_per_second:.1f} tok/s)")
            
            # Calculate aggregate metrics
            avg_time = total_time / len(self.test_prompts)
            avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            # Memory usage
            if device.type == "cuda":
                torch.cuda.synchronize()
                memory_used = torch.cuda.memory_allocated() / 1024**3
            else:
                memory_used = 0
            
            return {
                "framework": "pytorch",
                "device": str(device),
                "dtype": str(dtype),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "memory_usage_gb": memory_used,
                "total_time": total_time,
                "avg_inference_time": avg_time,
                "avg_tokens_per_second": avg_tokens_per_sec,
                "total_tokens_generated": total_tokens,
                "results": results,
                "success": True,
            }
            
        except Exception as e:
            print(f"  ❌ PyTorch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {"framework": "pytorch", "error": str(e), "success": False}
    
    def _manual_generate_pytorch(self, model, input_ids, max_new_tokens, temperature, device):
        """Manual generation fallback for PyTorch."""
        current_ids = input_ids
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(current_ids)
                logits = outputs.logits[:, -1, :] / temperature
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        return current_ids
    
    def load_jax_model(self):
        """Load the converted JAX model."""
        print("\n🔧 Loading converted JAX DiffuCoder model...")
        
        if not JAX_AVAILABLE:
            return None, None, None
        
        try:
            # Load config
            config_path = "./models/dream-jax/config.json"
            with open(config_path) as f:
                config_dict = json.load(f)
            
            print(f"  Config loaded: {config_dict['hidden_size']}d hidden, {config_dict['num_hidden_layers']} layers")
            
            # Create JAX config
            config = DreamConfig(
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
            print("  Creating JAX model...")
            model = DreamForCausalLM(config=config, dtype=jnp.bfloat16)
            
            # Load weights
            print("  Loading JAX weights...")
            import pickle
            start_time = time.time()
            
            params_path = "./models/dream-jax/params.pkl"
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            
            load_time = time.time() - start_time
            
            # Count parameters
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            print(f"  ✅ JAX model loaded in {load_time:.2f}s")
            print(f"     Parameters: {num_params/1e9:.1f}B")
            
            # Load tokenizer (same as PyTorch)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "./models/diffucoder-7b-complete",
                trust_remote_code=True,
                local_files_only=True,
            )
            
            return model, params, tokenizer
            
        except Exception as e:
            print(f"  ❌ Failed to load JAX model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def benchmark_jax(self, model, params, tokenizer) -> Dict[str, Any]:
        """Benchmark JAX inference."""
        if not JAX_AVAILABLE or model is None:
            return {"error": "JAX not available or model not loaded"}
        
        print("\n⚡ Benchmarking JAX inference...")
        
        try:
            # Initialize RNG
            rng = random.PRNGKey(42)
            
            # JIT compile the forward function
            @jax.jit
            def forward_fn(params, input_ids):
                return model.apply(params, input_ids, deterministic=True)
            
            # JIT compile generation step
            @jax.jit
            def generate_step(params, input_ids, rng_key, temperature):
                outputs = model.apply(params, input_ids, deterministic=True)
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Sample next token
                next_token = random.categorical(rng_key, logits)
                # Ensure correct shape for concatenation
                return jnp.expand_dims(next_token, axis=0)
            
            results = []
            total_time = 0
            total_tokens = 0
            
            # Warmup
            print("  Warming up...")
            dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
            _ = forward_fn(params, dummy_input)
            
            # Benchmark each prompt
            for i, prompt in enumerate(self.test_prompts):
                print(f"  [{i+1}/{len(self.test_prompts)}] Testing: {prompt[:30]}...")
                
                # Tokenize
                tokens = tokenizer.encode(prompt, max_length=512, truncation=True)
                input_ids = jnp.array([tokens])
                input_length = len(tokens)
                
                # Time generation
                start_time = time.time()
                
                # Generate tokens
                current_ids = input_ids
                generated_tokens = []
                
                for j in range(self.gen_params["max_new_tokens"]):
                    rng, step_rng = random.split(rng)
                    next_token = generate_step(
                        params, current_ids, step_rng, 
                        self.gen_params["temperature"]
                    )
                    current_ids = jnp.concatenate([current_ids, next_token], axis=-1)
                    generated_tokens.append(int(next_token[0, 0]))
                
                # Wait for computation to complete
                current_ids.block_until_ready()
                end_time = time.time()
                
                # Calculate metrics
                output_length = len(generated_tokens)
                inference_time = end_time - start_time
                tokens_per_second = output_length / inference_time if inference_time > 0 else 0
                
                # Decode output
                full_tokens = tokens + generated_tokens
                generated_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
                completion = generated_text[len(prompt):]
                
                results.append({
                    "prompt": prompt,
                    "completion": completion[:100] + "..." if len(completion) > 100 else completion,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                })
                
                total_time += inference_time
                total_tokens += output_length
                
                print(f"    Generated {output_length} tokens in {inference_time:.3f}s ({tokens_per_second:.1f} tok/s)")
            
            # Calculate aggregate metrics
            avg_time = total_time / len(self.test_prompts)
            avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            # Model info
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            
            return {
                "framework": "jax",
                "device": str(jax.devices()[0]),
                "model_parameters": num_params,
                "total_time": total_time,
                "avg_inference_time": avg_time,
                "avg_tokens_per_second": avg_tokens_per_sec,
                "total_tokens_generated": total_tokens,
                "results": results,
                "success": True,
            }
            
        except Exception as e:
            print(f"  ❌ JAX benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {"framework": "jax", "error": str(e), "success": False}
    
    def run_comparison(self):
        """Run full comparison benchmark."""
        print("🚀 Real DiffuCoder PyTorch vs JAX Inference Benchmark")
        print("=" * 70)
        
        # Load and benchmark PyTorch
        pytorch_model, pytorch_tokenizer, pytorch_config = self.load_pytorch_model()
        if pytorch_model is not None:
            self.results["pytorch_results"] = self.benchmark_pytorch(
                pytorch_model, pytorch_tokenizer, pytorch_config
            )
            # Free memory
            del pytorch_model
            gc.collect()
            if PYTORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self.results["pytorch_results"] = {"error": "Failed to load PyTorch model", "success": False}
        
        # Load and benchmark JAX
        jax_model, jax_params, jax_tokenizer = self.load_jax_model()
        if jax_model is not None:
            self.results["jax_results"] = self.benchmark_jax(
                jax_model, jax_params, jax_tokenizer
            )
        else:
            self.results["jax_results"] = {"error": "Failed to load JAX model", "success": False}
        
        # Compute comparison
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
        
        self.results["comparison"] = {
            "speed_comparison": {
                "pytorch_tokens_per_sec": pytorch_speed,
                "jax_tokens_per_sec": jax_speed,
                "jax_speedup": speed_ratio,
                "percentage_faster": (speed_ratio - 1) * 100 if speed_ratio > 1 else (1 - speed_ratio) * -100,
                "winner": "JAX" if speed_ratio > 1.0 else "PyTorch",
            },
            "time_comparison": {
                "pytorch_avg_time": pytorch_time,
                "jax_avg_time": jax_time,
                "time_ratio": time_ratio,
                "winner": "JAX" if time_ratio > 1.0 else "PyTorch",
            },
            "parameter_comparison": {
                "pytorch_params": pytorch_res.get("model_parameters", 0),
                "jax_params": jax_res.get("model_parameters", 0),
                "difference": abs(pytorch_res.get("model_parameters", 0) - jax_res.get("model_parameters", 0)),
            },
            "overall_winner": "JAX" if speed_ratio > 1.0 else "PyTorch",
            "summary": f"JAX is {speed_ratio:.2f}x faster" if speed_ratio > 1.0 else f"PyTorch is {1/speed_ratio:.2f}x faster"
        }
    
    def print_results(self):
        """Print formatted results."""
        print("\n" + "="*70)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        # System info
        print(f"\n🖥️  System Information:")
        print(f"   Platform: {self.results['system_info']['platform']}")
        print(f"   Python: {self.results['system_info']['python_version']}")
        
        if "pytorch" in self.results['system_info']:
            pt_info = self.results['system_info']['pytorch']
            print(f"   PyTorch: {pt_info['version']} (CUDA: {pt_info['cuda_available']}, MPS: {pt_info.get('mps_available', False)})")
        
        if "jax" in self.results['system_info']:
            jax_info = self.results['system_info']['jax']
            print(f"   JAX: {jax_info['version']} (Devices: {', '.join(jax_info['devices'])})")
        
        # PyTorch results
        print(f"\n🔥 PyTorch Results:")
        pytorch_res = self.results["pytorch_results"]
        if pytorch_res.get("success"):
            print(f"   Device: {pytorch_res['device']}")
            print(f"   Model parameters: {pytorch_res['model_parameters']:,}")
            print(f"   Average tokens/sec: {pytorch_res['avg_tokens_per_second']:.1f}")
            print(f"   Average time/prompt: {pytorch_res['avg_inference_time']:.3f}s")
            if pytorch_res.get('memory_usage_gb', 0) > 0:
                print(f"   GPU memory usage: {pytorch_res['memory_usage_gb']:.2f} GB")
        else:
            print(f"   ❌ Failed: {pytorch_res.get('error', 'Unknown error')}")
        
        # JAX results
        print(f"\n⚡ JAX Results:")
        jax_res = self.results["jax_results"]
        if jax_res.get("success"):
            print(f"   Device: {jax_res['device']}")
            print(f"   Model parameters: {jax_res['model_parameters']:,}")
            print(f"   Average tokens/sec: {jax_res['avg_tokens_per_second']:.1f}")
            print(f"   Average time/prompt: {jax_res['avg_inference_time']:.3f}s")
        else:
            print(f"   ❌ Failed: {jax_res.get('error', 'Unknown error')}")
        
        # Comparison
        if "comparison" in self.results and "error" not in self.results["comparison"]:
            comp = self.results["comparison"]
            print(f"\n📈 Performance Comparison:")
            print(f"   Speed winner: {comp['speed_comparison']['winner']}")
            print(f"   JAX speedup: {comp['speed_comparison']['jax_speedup']:.2f}x")
            if comp['speed_comparison']['percentage_faster'] > 0:
                print(f"   JAX is {comp['speed_comparison']['percentage_faster']:.1f}% faster")
            else:
                print(f"   PyTorch is {-comp['speed_comparison']['percentage_faster']:.1f}% faster")
            print(f"   Overall: {comp['summary']}")
        
        print("\n" + "="*70)
    
    def save_results(self, filename: str = "real_pytorch_vs_jax_results.json"):
        """Save results to file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📁 Results saved to {filename}")


def main():
    """Main function."""
    try:
        benchmark = RealModelBenchmark()
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