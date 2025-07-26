#!/usr/bin/env python3
"""
Real inference benchmark for DiffuCoder using simplified approach.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Set up environment
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import torch
    import transformers
    from transformers import AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
    PYTORCH_AVAILABLE = True
    print(f"✅ PyTorch {torch.__version__} available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch not available")

class SimpleBenchmarkRunner:
    """Simplified benchmark runner for DiffuCoder."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.results = {
            "model_path": str(self.model_path),
            "system_info": self._get_system_info(),
        }
        
        # Code generation test prompts
        self.test_prompts = [
            "def fibonacci(n):",
            "class Stack:",
            "def binary_search(arr, target):",
            "def quicksort(lst):",
            "def is_prime(n):",
        ]
    
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
                info["torch_device_name"] = torch.cuda.get_device_name(0)
                info["torch_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info
    
    def load_model_direct(self):
        """Load model using direct transformers approach."""
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        print("📥 Loading DiffuCoder model...")
        start_time = time.time()
        
        # Load tokenizer first
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=False,
        )
        print(f"    Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Import the model manually to avoid auto-loading issues
        print("  Setting up model imports...")
        model_dir = str(self.model_path.absolute())
        sys.path.insert(0, model_dir)
        
        # Import the model classes - fix relative imports
        try:
            from modeling_dream import DreamForCausalLM
            from configuration_dream import DreamConfig
        except ImportError as e:
            # Handle relative import issues by modifying the files temporarily
            print(f"    Import error: {e}")
            print("    Attempting to fix relative imports...")
            
            # Read and modify the modeling file to use absolute imports
            model_file = self.model_path / "modeling_dream.py"
            with open(model_file, 'r') as f:
                content = f.read()
            
            # Replace relative import with absolute import
            content = content.replace(
                "from .configuration_dream import DreamConfig",
                "from configuration_dream import DreamConfig"
            )
            content = content.replace(
                "from .generation_utils import DreamGenerationMixin, DreamGenerationConfig",
                "from generation_utils import DreamGenerationMixin, DreamGenerationConfig"
            )
            
            # Write temporary file
            temp_model_file = self.model_path / "modeling_dream_fixed.py"
            with open(temp_model_file, 'w') as f:
                f.write(content)
            
            # Import from fixed file
            import importlib.util
            spec = importlib.util.spec_from_file_location("modeling_dream_fixed", temp_model_file)
            modeling_module = importlib.util.module_from_spec(spec)
            sys.modules["modeling_dream_fixed"] = modeling_module
            spec.loader.exec_module(modeling_module)
            
            DreamForCausalLM = modeling_module.DreamForCausalLM
            
            # Import config separately
            from configuration_dream import DreamConfig
        
        print("  Loading model configuration...")
        config = DreamConfig.from_pretrained(str(self.model_path))
        print(f"    Model config: {config.hidden_size}d, {config.num_hidden_layers} layers")
        
        print("  Loading model weights...")
        model = DreamForCausalLM.from_pretrained(
            str(self.model_path),
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",  # Force CPU for stability
        )
        
        load_time = time.time() - start_time
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        model_info = {
            "num_parameters": num_params,
            "num_parameters_billions": num_params / 1e9,
            "model_dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
        }
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"   Parameters: {num_params/1e9:.1f}B")
        print(f"   Device: {model_info['device']}")
        print(f"   Dtype: {model_info['model_dtype']}")
        
        return model, tokenizer, model_info, load_time
    
    def benchmark_generation(self, model, tokenizer, model_info, load_time):
        """Benchmark text generation."""
        print("\n⚡ Running generation benchmark...")
        
        # Generation parameters
        gen_params = {
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        generations = []
        total_tokens = 0
        total_time = 0
        
        # Test each prompt
        for i, prompt in enumerate(self.test_prompts):
            print(f"  Testing prompt {i+1}/{len(self.test_prompts)}: {prompt}")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            input_length = inputs["input_ids"].shape[1]
            
            # Generate
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    # Check if model has diffusion_generate method
                    if hasattr(model, 'diffusion_generate'):
                        print("    Using diffusion generation...")
                        outputs = model.diffusion_generate(
                            inputs["input_ids"],
                            max_new_tokens=gen_params["max_new_tokens"],
                            temperature=gen_params["temperature"],
                            top_p=gen_params["top_p"],
                            steps=32,  # Reduced for speed
                        )
                    else:
                        print("    Using standard generation...")
                        outputs = model.generate(
                            **inputs,
                            **gen_params,
                        )
                
                generation_time = time.time() - start_time
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = generated_text[len(prompt):].strip()
                
                # Calculate metrics
                output_length = outputs.shape[1] - input_length
                tokens_per_second = output_length / generation_time if generation_time > 0 else 0
                
                # Store results
                generation_result = {
                    "prompt": prompt,
                    "completion": completion,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "generation_time_seconds": generation_time,
                    "tokens_per_second": tokens_per_second,
                }
                
                generations.append(generation_result)
                total_tokens += output_length
                total_time += generation_time
                
                print(f"    Generated {output_length} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
                print(f"    Preview: {completion[:100]}...")
                
            except Exception as e:
                print(f"    ❌ Generation failed: {e}")
                generation_result = {
                    "prompt": prompt,
                    "error": str(e),
                    "input_tokens": input_length,
                    "output_tokens": 0,
                    "generation_time_seconds": 0,
                    "tokens_per_second": 0,
                }
                generations.append(generation_result)
        
        # Calculate overall metrics
        successful_gens = [g for g in generations if "error" not in g]
        if successful_gens:
            avg_time = np.mean([g["generation_time_seconds"] for g in successful_gens])
            avg_tokens_per_sec = np.mean([g["tokens_per_second"] for g in successful_gens])
            success_rate = len(successful_gens) / len(generations)
        else:
            avg_time = 0
            avg_tokens_per_sec = 0
            success_rate = 0
        
        benchmark_results = {
            "framework": "pytorch",
            "model_info": model_info,
            "load_time_seconds": load_time,
            "generation_metrics": {
                "num_prompts": len(self.test_prompts),
                "successful_generations": len(successful_gens),
                "success_rate": success_rate,
                "total_tokens_generated": total_tokens,
                "total_generation_time_seconds": total_time,
                "avg_generation_time_seconds": avg_time,
                "avg_tokens_per_second": avg_tokens_per_sec,
            },
            "generations": generations,
            "success": len(successful_gens) > 0,
        }
        
        print(f"\n📊 Generation Results:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average generation time: {avg_time:.2f}s")
        print(f"   Average tokens/second: {avg_tokens_per_sec:.1f}")
        print(f"   Total tokens generated: {total_tokens}")
        
        return benchmark_results
    
    def run_benchmark(self):
        """Run the complete benchmark."""
        print("🚀 DiffuCoder Real Inference Benchmark")
        print("=" * 50)
        
        if not self.model_path.exists():
            return {"error": f"Model not found at {self.model_path}"}
        
        try:
            # Load model
            model, tokenizer, model_info, load_time = self.load_model_direct()
            
            # Run generation benchmark
            benchmark_results = self.benchmark_generation(model, tokenizer, model_info, load_time)
            
            # Add to results
            self.results["benchmark_results"] = benchmark_results
            
            return self.results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "system_info": self.results["system_info"],
            }
    
    def save_results(self, output_file: str = "real_benchmark_results.json"):
        """Save results to file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 Results saved to {output_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real DiffuCoder inference benchmark")
    parser.add_argument(
        "--model-path",
        default="./models/diffucoder-7b-complete",
        help="Path to DiffuCoder model"
    )
    parser.add_argument(
        "--output",
        default="real_benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    try:
        runner = SimpleBenchmarkRunner(args.model_path)
        results = runner.run_benchmark()
        runner.save_results(args.output)
        
        # Print final summary
        if "benchmark_results" in results and results["benchmark_results"]["success"]:
            metrics = results["benchmark_results"]["generation_metrics"]
            print(f"\n✅ Benchmark completed successfully!")
            print(f"🎯 Final Results:")
            print(f"   Model: {results['model_path']}")
            print(f"   Success rate: {metrics['success_rate']:.1%}")
            print(f"   Average tokens/second: {metrics['avg_tokens_per_second']:.1f}")
            print(f"   Total tokens generated: {metrics['total_tokens_generated']}")
        else:
            print(f"\n❌ Benchmark failed or had no successful generations")
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark cancelled by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")

if __name__ == "__main__":
    main()