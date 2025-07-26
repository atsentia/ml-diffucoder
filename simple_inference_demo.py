#!/usr/bin/env python3
"""
Simple inference demo for DiffuCoder using basic transformers functionality.
"""

import os
import sys
import time
import json
from pathlib import Path

# Set environment to allow remote code
os.environ["TRANSFORMERS_OFFLINE"] = "0"

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"✅ PyTorch {torch.__version__} available")
except ImportError:
    print("❌ PyTorch/transformers not available")
    sys.exit(1)

def main():
    """Run simple inference demo."""
    model_path = "./models/diffucoder-7b-complete"
    
    print("🚀 DiffuCoder Simple Inference Demo")
    print("=" * 40)
    
    # Test prompts
    prompts = [
        "def fibonacci(n):",
        "class BinaryTree:",
        "def quicksort(arr):",
    ]
    
    try:
        print("📥 Loading model and tokenizer...")
        start_time = time.time()
        
        # Load with trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model Info:")
        print(f"   Parameters: {num_params/1e9:.1f}B")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Dtype: {next(model.parameters()).dtype}")
        
        # Test generation
        print(f"\n🔮 Testing generation...")
        
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- Test {i+1}: {prompt} ---")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            start_gen = time.time()
            
            with torch.no_grad():
                try:
                    # Try diffusion generation if available
                    if hasattr(model, 'diffusion_generate'):
                        print("Using diffusion generation...")
                        outputs = model.diffusion_generate(
                            inputs["input_ids"],
                            max_new_tokens=150,
                            temperature=0.7,
                            top_p=0.95,
                            steps=64,
                        )
                        generation_method = "diffusion"
                    else:
                        print("Using standard generation...")
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=150,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        generation_method = "standard"
                    
                    gen_time = time.time() - start_gen
                    
                    # Decode
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    completion = generated_text[len(prompt):].strip()
                    
                    # Metrics
                    output_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                    tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0
                    
                    print(f"✅ Generated {output_tokens} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
                    print(f"Method: {generation_method}")
                    print(f"Completion:\n{completion}")
                    
                    results.append({
                        "prompt": prompt,
                        "completion": completion,
                        "generation_method": generation_method,
                        "generation_time": gen_time,
                        "output_tokens": output_tokens,
                        "tokens_per_second": tokens_per_sec,
                        "success": True,
                    })
                    
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    results.append({
                        "prompt": prompt,
                        "error": str(e),
                        "success": False,
                    })
        
        # Save results
        demo_results = {
            "model_path": model_path,
            "load_time_seconds": load_time,
            "model_info": {
                "num_parameters": num_params,
                "num_parameters_billions": num_params / 1e9,
            },
            "results": results,
            "system_info": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
        }
        
        with open("simple_inference_results.json", "w") as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        # Summary
        successful = [r for r in results if r.get("success", False)]
        print(f"\n📊 Demo Summary:")
        print(f"   Successful generations: {len(successful)}/{len(results)}")
        
        if successful:
            avg_tokens_per_sec = sum(r["tokens_per_second"] for r in successful) / len(successful)
            print(f"   Average tokens/second: {avg_tokens_per_sec:.1f}")
            print(f"   Total tokens generated: {sum(r['output_tokens'] for r in successful)}")
        
        print(f"\n📁 Results saved to simple_inference_results.json")
        print("✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()