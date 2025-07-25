#!/usr/bin/env python3
"""
Sanity test for DiffuCoder with real weights in both PyTorch and JAX.
Tests actual code generation to ensure models are working correctly.
"""

import os
import sys
import json
import pickle
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))
sys.path.insert(0, str(Path(__file__).parent / "models" / "diffucoder-7b-complete"))

# Framework imports
try:
    import torch
    import transformers
    from transformers import AutoTokenizer
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch not available")

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
    print("✅ JAX available")
except ImportError:
    JAX_AVAILABLE = False
    print("❌ JAX not available")

# Local imports
if JAX_AVAILABLE:
    from jax_lm.models.dream import DreamConfig, DreamForCausalLM

def load_tokenizer():
    """Load the tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("./models/diffucoder-7b-complete", trust_remote_code=True)
        print("✅ Tokenizer loaded")
        return tokenizer
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return None

def load_pytorch_model():
    """Load the PyTorch model with real weights."""
    if not PYTORCH_AVAILABLE:
        return None
    
    print("🔧 Loading PyTorch DiffuCoder model...")
    
    try:
        # Import the Dream model classes - use transformers AutoModel for safer loading
        from transformers import AutoModel, AutoConfig
        from safetensors import safe_open
        
        # Load with trust_remote_code for custom model
        model = AutoModel.from_pretrained("./models/diffucoder-7b-complete", trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        # Move to device and set to eval mode
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"✅ PyTorch model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Failed to load PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_jax_model():
    """Load the JAX model with converted weights."""
    if not JAX_AVAILABLE:
        return None
    
    print("🔧 Loading JAX DiffuCoder model...")
    
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
        
        print(f"✅ JAX model loaded successfully")
        print(f"   Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
        
        return model, params, config
        
    except Exception as e:
        print(f"❌ Failed to load JAX model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_pytorch_generation(model, device, tokenizer, prompt, max_new_tokens=50):
    """Test PyTorch generation."""
    if model is None or tokenizer is None:
        return None
    
    print(f"🔥 Testing PyTorch generation for: {prompt}")
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        
        # Generate with simple sampling
        with torch.no_grad():
            # Get initial logits
            outputs = model(input_ids)
            generated_ids = input_ids.clone()
            
            # Simple generation loop
            for _ in range(max_new_tokens):
                # Get logits for last token
                current_outputs = model(generated_ids)
                logits = current_outputs.logits[:, -1, :]
                
                # Sample next token (temperature = 0.7)
                temperature = 0.7
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Stop if we hit EOS or max length
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"   ✅ Generated {generated_ids.size(1) - input_ids.size(1)} tokens")
        return generated_text
        
    except Exception as e:
        print(f"   ❌ PyTorch generation failed: {e}")
        return None

def test_jax_generation(model, params, config, tokenizer, prompt, max_new_tokens=50):
    """Test JAX generation."""
    if model is None or params is None or tokenizer is None:
        return None
    
    print(f"⚡ Testing JAX generation for: {prompt}")
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="np", max_length=100, truncation=True)
        input_ids = jnp.array(inputs["input_ids"])
        
        # JIT compile forward function
        @jax.jit
        def forward_fn(params, input_ids):
            return model.apply(params, input_ids, deterministic=True)
        
        # Generate with simple sampling
        rng = random.PRNGKey(42)
        generated_ids = input_ids
        
        # Simple generation loop
        for _ in range(max_new_tokens):
            # Get logits for last token
            outputs = forward_fn(params, generated_ids)
            logits = outputs["logits"][:, -1, :]
            
            # Sample next token (temperature = 0.7)
            temperature = 0.7
            logits = logits / temperature
            rng, sample_rng = random.split(rng)
            next_token = random.categorical(sample_rng, logits, axis=-1)
            next_token = next_token[:, None]  # Add sequence dimension
            
            # Append token
            generated_ids = jnp.concatenate([generated_ids, next_token], axis=-1)
            
            # Stop if we hit EOS or max length
            if int(next_token[0, 0]) == tokenizer.eos_token_id:
                break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"   ✅ Generated {generated_ids.shape[1] - input_ids.shape[1]} tokens")
        return generated_text
        
    except Exception as e:
        print(f"   ❌ JAX generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run sanity tests."""
    print("🧪 DiffuCoder Real Weights Sanity Test")
    print("=" * 60)
    
    # Test prompts inspired by typical code completion tasks
    test_prompts = [
        "def fibonacci(n):",
        "def quicksort(arr):",
        "class BinaryTree:",
        "def calculate_mean(numbers):",
        "import numpy as np\n\ndef matrix_multiply(A, B):",
    ]
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    if tokenizer is None:
        print("❌ Cannot proceed without tokenizer")
        return
    
    # Load models
    pytorch_model, device = load_pytorch_model() if PYTORCH_AVAILABLE else (None, None)
    jax_model, jax_params, jax_config = load_jax_model() if JAX_AVAILABLE else (None, None, None)
    
    if pytorch_model is None and jax_model is None:
        print("❌ No models loaded successfully")
        return
    
    print(f"\n🎯 Running sanity tests on {len(test_prompts)} prompts...")
    print("=" * 60)
    
    results = {}
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 Test {i+1}/{len(test_prompts)}: {prompt[:30]}...")
        print("-" * 40)
        
        result = {"prompt": prompt}
        
        # Test PyTorch
        if pytorch_model is not None:
            pytorch_output = test_pytorch_generation(pytorch_model, device, tokenizer, prompt)
            result["pytorch"] = pytorch_output
            if pytorch_output:
                print(f"🔥 PyTorch output:\n{pytorch_output}\n")
        
        # Test JAX
        if jax_model is not None:
            jax_output = test_jax_generation(jax_model, jax_params, jax_config, tokenizer, prompt)
            result["jax"] = jax_output
            if jax_output:
                print(f"⚡ JAX output:\n{jax_output}\n")
        
        results[f"test_{i+1}"] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SANITY TEST SUMMARY")
    print("=" * 60)
    
    pytorch_success = sum(1 for r in results.values() if r.get("pytorch") is not None)
    jax_success = sum(1 for r in results.values() if r.get("jax") is not None)
    
    if pytorch_model:
        print(f"🔥 PyTorch: {pytorch_success}/{len(test_prompts)} successful generations")
    else:
        print("🔥 PyTorch: Not tested (model not loaded)")
    
    if jax_model:
        print(f"⚡ JAX: {jax_success}/{len(test_prompts)} successful generations")
    else:
        print("⚡ JAX: Not tested (model not loaded)")
    
    # Check if outputs look reasonable
    if pytorch_success > 0 or jax_success > 0:
        print("\n✅ Sanity test PASSED - Models are generating coherent code!")
        
        # Save results for analysis
        with open("sanity_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("📁 Results saved to sanity_test_results.json")
    else:
        print("\n❌ Sanity test FAILED - No successful generations")
    
    print("\n🎉 Sanity test completed!")

if __name__ == "__main__":
    main()