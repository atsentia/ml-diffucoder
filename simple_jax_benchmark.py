#!/usr/bin/env python3
"""
Simple JAX inference benchmark using real converted DiffuCoder weights.
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

# Add jax_lm to path
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

from jax_lm.models.dream import DreamConfig, DreamForCausalLM

def load_real_jax_model():
    """Load the real converted JAX model."""
    print("🔧 Loading real JAX DiffuCoder model...")
    
    try:
        # Load configuration
        config_file = "./models/dream-jax/config.json"
        with open(config_file) as f:
            config_dict = json.load(f)
        
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
        model = DreamForCausalLM(config=config, dtype=config.dtype)
        
        # Load real weights
        params_file = "./models/dream-jax/params.pkl"
        with open(params_file, "rb") as f:
            params = pickle.load(f)
        
        print(f"✅ Real JAX model loaded successfully")
        print(f"   Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
        print(f"   Config: {config.num_hidden_layers} layers, {config.hidden_size}d, {config.vocab_size} vocab")
        
        return model, params, config
        
    except Exception as e:
        print(f"❌ Failed to load real JAX model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_inference_benchmark(model, params, config, num_trials=5):
    """Run inference benchmark with the real model."""
    print(f"\n⚡ Running JAX inference benchmark ({num_trials} trials)...")
    
    # Test prompts
    test_prompts = [
        "def fibonacci(n):",
        "class BinaryTree:",
        "def quicksort(arr):",
        "def binary_search(arr, target):",
        "import numpy as np\n\ndef matrix_multiply(A, B):",
    ]
    
    # Simple tokenizer (character-level for demo)
    def simple_tokenize(text):
        return jnp.array([[ord(c) % min(1000, config.vocab_size) for c in text[:20]]])
    
    # JIT compile the forward function
    @jax.jit
    def forward_fn(params, input_ids):
        return model.apply(params, input_ids, deterministic=True)
    
    # Warmup
    print("   Warming up JIT compilation...")
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    _ = forward_fn(params, dummy_input)
    print("   ✅ JIT compilation complete")
    
    results = []
    total_time = 0
    
    for trial in range(num_trials):
        print(f"\n   Trial {trial + 1}/{num_trials}:")
        trial_time = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"     Testing: {prompt[:30]}...")
            
            # Tokenize
            input_ids = simple_tokenize(prompt)
            input_length = input_ids.shape[1]
            
            # Time forward pass
            start_time = time.time()
            
            # Run inference
            output = forward_fn(params, input_ids)
            
            # Make sure computation is complete
            logits = output["logits"]
            _ = jnp.sum(logits).block_until_ready()
            
            end_time = time.time()
            
            inference_time = end_time - start_time
            trial_time += inference_time
            
            print(f"       Input: {input_length} tokens, Time: {inference_time:.4f}s")
            print(f"       Output shape: {logits.shape}")
            
            results.append({
                "trial": trial + 1,
                "prompt_idx": i,
                "prompt": prompt,
                "input_tokens": input_length,
                "inference_time": inference_time,
                "output_shape": logits.shape,
            })
        
        total_time += trial_time
        print(f"     Trial {trial + 1} total time: {trial_time:.4f}s")
    
    avg_time = total_time / (num_trials * len(test_prompts))
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Total trials: {num_trials}")
    print(f"   Total prompts: {len(test_prompts)}")
    print(f"   Total time: {total_time:.4f}s")
    print(f"   Average time per inference: {avg_time:.4f}s")
    print(f"   Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    
    return {
        "num_trials": num_trials,
        "num_prompts": len(test_prompts),
        "total_time": total_time,
        "avg_time_per_inference": avg_time,
        "results": results,
        "model_info": {
            "parameters": sum(x.size for x in jax.tree_util.tree_leaves(params)),
            "layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
        }
    }


def main():
    """Main benchmark function."""
    print("🚀 JAX DiffuCoder Real Weights Inference Benchmark")
    print("=" * 60)
    
    # Check JAX devices
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    
    # Load model
    model, params, config = load_real_jax_model()
    if model is None:
        print("❌ Failed to load model, exiting")
        return
    
    # Run benchmark
    results = run_inference_benchmark(model, params, config)
    
    # Save results
    output_file = "jax_real_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Benchmark completed!")
    print(f"📁 Results saved to {output_file}")


if __name__ == "__main__":
    main()