#!/usr/bin/env python3
"""
Quick sanity test showing forward pass outputs from both models.
"""

import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "jax_lm"))

try:
    import torch
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
    print("✅ JAX available")
    
    from jax_lm.models.dream import DreamConfig, DreamForCausalLM
except ImportError:
    JAX_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/diffucoder-7b-complete", trust_remote_code=True)
    print("✅ Tokenizer loaded")
except Exception as e:
    print(f"❌ Tokenizer failed: {e}")
    tokenizer = None

def test_jax_forward_pass():
    """Test JAX model forward pass."""
    print("\n⚡ Testing JAX forward pass...")
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available")
        return
    
    try:
        # Load JAX model
        with open("./models/dream-jax/config.json") as f:
            config_dict = json.load(f)
        
        config = DreamConfig(**{k: v for k, v in config_dict.items() if k in DreamConfig.__dataclass_fields__})
        model = DreamForCausalLM(config=config, dtype=config.dtype)
        
        # Load weights
        with open("./models/dream-jax/params.pkl", "rb") as f:
            params = pickle.load(f)
        
        print(f"✅ JAX model loaded: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
        
        # Test prompts
        prompts = [
            "def fibonacci(n):",
            "def quicksort(arr):",
            "class BinaryTree:",
        ]
        
        @jax.jit
        def forward_fn(params, input_ids):
            return model.apply(params, input_ids, deterministic=True)
        
        for prompt in prompts:
            print(f"\n   Testing: {prompt}")
            
            if tokenizer:
                # Use real tokenizer
                inputs = tokenizer(prompt, return_tensors="np", max_length=50, truncation=True)
                input_ids = jnp.array(inputs["input_ids"])
            else:
                # Fallback: simple character tokenization
                input_ids = jnp.array([[ord(c) % min(1000, config.vocab_size) for c in prompt[:20]]])
            
            # Forward pass
            outputs = forward_fn(params, input_ids)
            logits = outputs["logits"]
            
            print(f"     Input shape: {input_ids.shape}")
            print(f"     Output shape: {logits.shape}")
            print(f"     Logits range: {float(logits.min()):.3f} to {float(logits.max()):.3f}")
            
            # Get top-k predictions for last token
            last_logits = logits[0, -1, :]
            top_k = 5
            top_indices = jnp.argsort(last_logits)[-top_k:]
            top_probs = jax.nn.softmax(last_logits)[top_indices]
            
            print(f"     Top {top_k} next token predictions:")
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                if tokenizer:
                    token = tokenizer.decode([int(idx)])
                    print(f"       {i+1}. '{token}' (prob: {float(prob):.4f})")
                else:
                    print(f"       {i+1}. Token {int(idx)} (prob: {float(prob):.4f})")
        
        print("✅ JAX forward pass successful!")
        
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        import traceback
        traceback.print_exc()

def test_pytorch_forward_pass():
    """Test PyTorch model forward pass."""
    print("\n🔥 Testing PyTorch forward pass...")
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return
    
    try:
        from transformers import AutoModel
        
        # Load PyTorch model (this may take time for the full 7B model)
        print("   Loading model (this may take a moment)...")
        model = AutoModel.from_pretrained(
            "./models/diffucoder-7b-complete", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"✅ PyTorch model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"   Device: {device}")
        
        prompts = [
            "def fibonacci(n):",
            "def quicksort(arr):",
            "class BinaryTree:",
        ]
        
        for prompt in prompts:
            print(f"\n   Testing: {prompt}")
            
            if tokenizer:
                # Use real tokenizer
                inputs = tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
                input_ids = inputs["input_ids"].to(device)
            else:
                # Fallback: simple character tokenization
                vocab_size = getattr(model.config, 'vocab_size', 50000)
                input_ids = torch.tensor([[ord(c) % min(1000, vocab_size) for c in prompt[:20]]], device=device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'last_hidden_state'):
                    # If it's just the base model, we won't get logits
                    print(f"     Hidden state shape: {outputs.last_hidden_state.shape}")
                    print("     (Note: Base model - no logits, would need language model head)")
                    continue
                else:
                    print(f"     Output type: {type(outputs)}")
                    continue
            
            print(f"     Input shape: {input_ids.shape}")
            print(f"     Output shape: {logits.shape}")
            print(f"     Logits range: {float(logits.min()):.3f} to {float(logits.max()):.3f}")
            
            # Get top-k predictions for last token
            last_logits = logits[0, -1, :]
            top_k = 5
            top_values, top_indices = torch.topk(last_logits, top_k)
            top_probs = torch.softmax(last_logits, dim=-1)[top_indices]
            
            print(f"     Top {top_k} next token predictions:")
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                if tokenizer:
                    token = tokenizer.decode([int(idx)])
                    print(f"       {i+1}. '{token}' (prob: {float(prob):.4f})")
                else:
                    print(f"       {i+1}. Token {int(idx)} (prob: {float(prob):.4f})")
        
        print("✅ PyTorch forward pass successful!")
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run quick sanity tests."""
    print("🧪 Quick DiffuCoder Sanity Test")
    print("=" * 50)
    
    # Test JAX first (lighter weight)
    test_jax_forward_pass()
    
    # Test PyTorch (heavier)
    test_pytorch_forward_pass()
    
    print("\n" + "=" * 50)
    print("🎉 Quick sanity test completed!")
    print("\nNote: This tests forward pass and next-token prediction.")
    print("For full generation, see the complete benchmark scripts.")

if __name__ == "__main__":
    main()