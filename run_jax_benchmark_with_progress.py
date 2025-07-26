#!/usr/bin/env python3
"""
JAX DiffuCoder benchmark with detailed progress logging.
Focuses on JAX performance with real weights.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

# Add jax_lm to path
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available, using simple progress logging")

# Framework imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    JAX_AVAILABLE = True
    print(f"✅ JAX {jax.__version__} available")
    print(f"   Devices: {[str(d) for d in jax.devices()]}")
except ImportError:
    JAX_AVAILABLE = False
    print("❌ JAX not available")
    sys.exit(1)

# Local imports
from jax_lm.models.dream import DreamConfig, DreamForCausalLM

# Also try to import transformers for tokenizer
try:
    from transformers import AutoTokenizer
    print("✅ Transformers available for tokenizer")
except ImportError:
    print("❌ Transformers not available")
    sys.exit(1)


class ProgressLogger:
    """Simple progress logger that prints updates."""
    
    def __init__(self, total_steps: int, desc: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.desc = desc
        self.start_time = time.time()
        
    def update(self, n: int = 1, postfix: str = ""):
        self.current_step += n
        elapsed = time.time() - self.start_time
        progress = self.current_step / self.total_steps
        
        # Calculate ETA
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "?"
        
        # Print progress
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r{self.desc}: [{bar}] {self.current_step}/{self.total_steps} "
              f"({progress*100:.1f}%) - {elapsed:.1f}s - ETA: {eta_str} {postfix}", 
              end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line at completion


class JAXBenchmarkWithProgress:
    """JAX benchmark with detailed progress logging."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "jax_results": {},
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
        
        print("\n📊 Benchmark Configuration:")
        print(f"   Test prompts: {len(self.test_prompts)}")
        print(f"   Max new tokens: {self.gen_params['max_new_tokens']}")
        print(f"   Temperature: {self.gen_params['temperature']}")
        print(f"   Sampling: {self.gen_params['do_sample']}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "jax": {
                "version": jax.__version__,
                "devices": [str(d) for d in jax.devices()],
                "device_count": jax.device_count(),
                "backend": jax.default_backend(),
            }
        }
        return info
    
    def _log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "WARNING": "⚠️ ",
            "ERROR": "❌",
            "PROGRESS": "⏳",
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")
    
    def load_jax_model(self):
        """Load the converted JAX model with progress updates."""
        self._log("Starting JAX model loading process...", "INFO")
        
        try:
            # Step 1: Load config
            self._log("Loading model configuration...", "PROGRESS")
            config_path = "./models/dream-jax/config.json"
            
            with open(config_path) as f:
                config_dict = json.load(f)
            
            self._log(f"Config loaded successfully:", "SUCCESS")
            self._log(f"  - Hidden size: {config_dict['hidden_size']}")
            self._log(f"  - Layers: {config_dict['num_hidden_layers']}")
            self._log(f"  - Attention heads: {config_dict['num_attention_heads']}")
            self._log(f"  - Vocab size: {config_dict['vocab_size']}")
            
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
            
            # Step 2: Create model
            self._log("Creating JAX model architecture...", "PROGRESS")
            model = DreamForCausalLM(config=config, dtype=jnp.bfloat16)
            self._log("Model architecture created", "SUCCESS")
            
            # Step 3: Load weights
            self._log("Loading model weights (this may take a while)...", "PROGRESS")
            import pickle
            start_time = time.time()
            
            params_path = "./models/dream-jax/params.pkl"
            file_size = os.path.getsize(params_path) / (1024**3)  # GB
            self._log(f"  Weight file size: {file_size:.2f} GB")
            
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            
            load_time = time.time() - start_time
            
            # Count parameters
            self._log("Counting model parameters...", "PROGRESS")
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            
            self._log(f"JAX model loaded successfully!", "SUCCESS")
            self._log(f"  - Load time: {load_time:.2f}s")
            self._log(f"  - Parameters: {num_params:,} ({num_params/1e9:.1f}B)")
            self._log(f"  - Weight loading speed: {file_size/load_time:.2f} GB/s")
            
            # Step 4: Load tokenizer
            self._log("Loading tokenizer...", "PROGRESS")
            tokenizer = AutoTokenizer.from_pretrained(
                "./models/diffucoder-7b-complete",
                trust_remote_code=True,
                local_files_only=True,
            )
            self._log(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})", "SUCCESS")
            
            return model, params, tokenizer
            
        except Exception as e:
            self._log(f"Failed to load JAX model: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def benchmark_jax_inference(self, model, params, tokenizer) -> Dict[str, Any]:
        """Benchmark JAX inference with detailed progress."""
        self._log("Starting JAX inference benchmark...", "INFO")
        
        try:
            # Initialize RNG
            rng = random.PRNGKey(42)
            
            # Step 1: JIT compilation
            self._log("JIT compiling model functions...", "PROGRESS")
            
            @jax.jit
            def forward_fn(params, input_ids):
                return model.apply(params, input_ids, deterministic=True)
            
            @jax.jit
            def generate_step(params, input_ids, rng_key, temperature):
                outputs = model.apply(params, input_ids, deterministic=True)
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Sample next token
                next_token = random.categorical(rng_key, logits)
                # Ensure correct shape for concatenation
                return jnp.expand_dims(next_token, axis=0)
            
            # Step 2: Warmup
            self._log("Running warmup iterations...", "PROGRESS")
            warmup_steps = 3
            
            if TQDM_AVAILABLE:
                warmup_bar = tqdm(range(warmup_steps), desc="Warmup", leave=False)
            else:
                warmup_bar = range(warmup_steps)
                
            for i in warmup_bar:
                dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
                _ = forward_fn(params, dummy_input)
                if not TQDM_AVAILABLE:
                    print(f"  Warmup {i+1}/{warmup_steps} complete")
            
            self._log("Warmup complete", "SUCCESS")
            
            # Step 3: Benchmark each prompt
            results = []
            total_time = 0
            total_tokens = 0
            
            self._log(f"Benchmarking {len(self.test_prompts)} prompts...", "INFO")
            
            if TQDM_AVAILABLE:
                prompt_bar = tqdm(enumerate(self.test_prompts), 
                                total=len(self.test_prompts), 
                                desc="Prompts")
            else:
                prompt_bar = enumerate(self.test_prompts)
            
            for i, prompt in prompt_bar:
                self._log(f"[Prompt {i+1}/{len(self.test_prompts)}] {prompt[:30]}...", "PROGRESS")
                
                # Tokenize
                tokens = tokenizer.encode(prompt, max_length=512, truncation=True)
                input_ids = jnp.array([tokens])
                input_length = len(tokens)
                
                # Progress for token generation
                if TQDM_AVAILABLE:
                    token_bar = tqdm(range(self.gen_params["max_new_tokens"]), 
                                   desc=f"  Generating", 
                                   leave=False)
                else:
                    progress = ProgressLogger(self.gen_params["max_new_tokens"], 
                                            desc=f"  Generating tokens")
                
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
                    
                    # Update progress
                    if TQDM_AVAILABLE:
                        token_bar.update(1)
                    else:
                        progress.update(1)
                
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
                
                result = {
                    "prompt": prompt,
                    "completion": completion[:100] + "..." if len(completion) > 100 else completion,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                }
                results.append(result)
                
                total_time += inference_time
                total_tokens += output_length
                
                self._log(f"  Generated {output_length} tokens in {inference_time:.3f}s ({tokens_per_second:.1f} tok/s)", "SUCCESS")
                self._log(f"  Preview: {completion[:50]}...", "INFO")
            
            # Calculate aggregate metrics
            self._log("Calculating final metrics...", "PROGRESS")
            
            avg_time = total_time / len(self.test_prompts)
            avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            # Model info
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            
            benchmark_results = {
                "framework": "jax",
                "device": str(jax.devices()[0]),
                "backend": jax.default_backend(),
                "model_parameters": num_params,
                "total_time": total_time,
                "avg_inference_time": avg_time,
                "avg_tokens_per_second": avg_tokens_per_sec,
                "total_tokens_generated": total_tokens,
                "results": results,
                "success": True,
            }
            
            self._log("Benchmark completed successfully!", "SUCCESS")
            
            return benchmark_results
            
        except Exception as e:
            self._log(f"JAX benchmark failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return {"framework": "jax", "error": str(e), "success": False}
    
    def run_benchmark(self):
        """Run the complete JAX benchmark."""
        print("\n" + "="*70)
        print("🚀 JAX DiffuCoder Real Weights Inference Benchmark")
        print("="*70)
        
        # Load model
        model, params, tokenizer = self.load_jax_model()
        
        if model is None:
            self.results["jax_results"] = {"error": "Failed to load model", "success": False}
            return self.results
        
        # Run benchmark
        print("\n" + "-"*70)
        self.results["jax_results"] = self.benchmark_jax_inference(model, params, tokenizer)
        
        return self.results
    
    def print_results(self):
        """Print formatted results."""
        print("\n" + "="*70)
        print("📊 JAX BENCHMARK RESULTS")
        print("="*70)
        
        jax_res = self.results["jax_results"]
        
        if jax_res.get("success"):
            print(f"\n⚡ JAX Performance:")
            print(f"   Device: {jax_res['device']}")
            print(f"   Backend: {jax_res['backend']}")
            print(f"   Model parameters: {jax_res['model_parameters']:,}")
            print(f"   Total generation time: {jax_res['total_time']:.2f}s")
            print(f"   Average tokens/sec: {jax_res['avg_tokens_per_second']:.1f}")
            print(f"   Average time/prompt: {jax_res['avg_inference_time']:.3f}s")
            print(f"   Total tokens generated: {jax_res['total_tokens_generated']}")
            
            print(f"\n📝 Individual Results:")
            for i, result in enumerate(jax_res['results']):
                print(f"\n   Prompt {i+1}: {result['prompt']}")
                print(f"   Tokens: {result['output_tokens']} in {result['inference_time']:.3f}s ({result['tokens_per_second']:.1f} tok/s)")
                print(f"   Preview: {result['completion'][:80]}...")
        else:
            print(f"\n⚡ JAX: ❌ Failed: {jax_res.get('error', 'Unknown error')}")
        
        print("\n" + "="*70)
    
    def save_results(self, filename: str = "jax_benchmark_results.json"):
        """Save results to file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        self._log(f"Results saved to {filename}", "SUCCESS")


def main():
    """Main function."""
    try:
        # Check if tqdm is available, if not, offer to install
        if not TQDM_AVAILABLE:
            print("\n💡 Tip: Install tqdm for better progress bars:")
            print("   pip install tqdm")
            print("   Continuing with simple progress logging...\n")
        
        benchmark = JAXBenchmarkWithProgress()
        results = benchmark.run_benchmark()
        benchmark.print_results()
        benchmark.save_results()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmark cancelled by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()