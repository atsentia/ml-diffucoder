#!/usr/bin/env python3
"""
Example script for inference with GRPO-trained DiffuCoder models in JAX.

This script demonstrates how to load a GRPO-trained model and generate code
completions using diffusion-based generation.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add the jax_lm package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.utils import load_model
from jax_lm.utils.tokenizer import load_tokenizer


class GRPOInference:
    """Inference wrapper for GRPO-trained DiffuCoder models."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        dtype: str = "bfloat16",
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_path: Path to tokenizer (defaults to model_path)
            dtype: Model dtype for inference
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = tokenizer_path or model_path
        self.dtype = self._get_dtype(dtype)
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
        # Initialize random key
        self.rng = random.PRNGKey(42)
    
    def _get_dtype(self, dtype_str: str):
        """Convert dtype string to JAX dtype."""
        dtype_map = {
            "float32": jnp.float32,
            "float16": jnp.float16,
            "bfloat16": jnp.bfloat16,
        }
        return dtype_map.get(dtype_str, jnp.float32)
    
    def _load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}")
        
        try:
            self.model, self.params = load_model(
                str(self.model_path),
                dtype=self.dtype,
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        print(f"Loading tokenizer from {self.tokenizer_path}")
        
        try:
            self.tokenizer = load_tokenizer(self.tokenizer_path)
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise
    
    def generate_code(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        diffusion_steps: int = 64,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate code completions for a given prompt.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            diffusion_steps: Number of diffusion steps
            num_samples: Number of samples to generate
            
        Returns:
            List of generated code completions
        """
        print(f"Generating {num_samples} completion(s) for prompt:")
        print(f"  '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs["attention_mask"])
        
        # Expand for multiple samples
        if num_samples > 1:
            input_ids = jnp.tile(input_ids, (num_samples, 1))
            attention_mask = jnp.tile(attention_mask, (num_samples, 1))
        
        completions = []
        
        for i in range(num_samples):
            # Split random key
            self.rng, gen_rng = random.split(self.rng)
            
            try:
                # Generate completion using diffusion
                # Note: This is a simplified version - actual implementation would
                # use the diffusion generation method from the model
                output = self._diffusion_generate(
                    input_ids[i:i+1] if num_samples > 1 else input_ids,
                    attention_mask[i:i+1] if num_samples > 1 else attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    diffusion_steps=diffusion_steps,
                    rng=gen_rng,
                )
                
                # Decode the generated tokens
                generated_tokens = output["sequences"]
                generated_text = self.tokenizer.decode(
                    generated_tokens[0], 
                    skip_special_tokens=True
                )
                
                # Remove the original prompt from the completion
                completion = generated_text[len(prompt):].strip()
                completions.append(completion)
                
            except Exception as e:
                print(f"⚠️  Failed to generate sample {i+1}: {e}")
                completions.append(f"[Generation failed: {e}]")
        
        return completions
    
    def _diffusion_generate(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        diffusion_steps: int,
        rng: random.PRNGKey,
    ):
        """
        Simplified diffusion generation (placeholder implementation).
        
        In practice, this would implement the full diffusion generation process
        with iterative denoising and masked token prediction.
        """
        batch_size, seq_len = input_ids.shape
        
        # For demonstration, we'll do a simple forward pass
        # Real implementation would do iterative diffusion sampling
        outputs = self.model.apply(
            self.params,
            input_ids,
            attention_mask=attention_mask,
            deterministic=True,
        )
        
        logits = outputs["logits"]
        
        # Simple greedy decoding for demonstration
        # Real implementation would use proper diffusion sampling
        generated_ids = jnp.argmax(logits, axis=-1)
        
        # Concatenate with original input
        sequences = jnp.concatenate([
            input_ids,
            generated_ids[:, :max_new_tokens]
        ], axis=-1)
        
        return {"sequences": sequences}


def demo_code_generation():
    """Demonstrate code generation with various prompts."""
    
    # Example prompts for code generation
    prompts = [
        "def fibonacci(n):",
        "class BinaryTree:",
        "# Function to sort a list using quicksort\ndef quicksort(arr):",
        "import numpy as np\n\ndef matrix_multiply(A, B):",
        "# Create a decorator for timing function execution\ndef timer(func):",
    ]
    
    # Model path (adjust as needed)
    model_path = "./grpo_checkpoints"  # Path to your trained model
    
    try:
        # Initialize inference pipeline
        print("🔧 Initializing GRPO inference pipeline...")
        inference = GRPOInference(
            model_path=model_path,
            dtype="bfloat16",
        )
        
        print(f"\n🖥️  Using JAX devices: {jax.devices()}")
        
        # Generate completions for each prompt
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*60}")
            print(f"Example {i}/{len(prompts)}")
            print(f"{'='*60}")
            
            # Generate multiple completions
            completions = inference.generate_code(
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.7,
                diffusion_steps=32,  # Faster for demo
                num_samples=2,
            )
            
            print(f"\n📝 Prompt:")
            print(f"```python\n{prompt}\n```")
            
            for j, completion in enumerate(completions, 1):
                print(f"\n🔮 Completion {j}:")
                print(f"```python\n{prompt}{completion}\n```")
        
        print(f"\n✅ Demo completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("Please ensure you have trained a model using grpo_training.py first")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


def interactive_mode():
    """Interactive mode for testing prompts."""
    
    model_path = "./grpo_checkpoints"
    
    try:
        print("🔧 Initializing interactive GRPO inference...")
        inference = GRPOInference(model_path=model_path)
        
        print("\n🎮 Interactive Mode Started!")
        print("Enter your code prompts (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\n📝 Prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                # Generate completion
                completions = inference.generate_code(
                    prompt=prompt,
                    max_new_tokens=150,
                    temperature=0.6,
                    num_samples=1,
                )
                
                print(f"\n🔮 Generated completion:")
                print("```python")
                print(prompt + completions[0])
                print("```")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"⚠️  Error: {e}")
        
    except Exception as e:
        print(f"❌ Failed to start interactive mode: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GRPO-trained DiffuCoder inference")
    parser.add_argument(
        "--mode", 
        choices=["demo", "interactive"], 
        default="demo",
        help="Inference mode"
    )
    parser.add_argument(
        "--model-path",
        default="./grpo_checkpoints",
        help="Path to trained model checkpoint"
    )
    
    args = parser.parse_args()
    
    print("🚀 JAX DiffuCoder GRPO Inference")
    print("=" * 40)
    
    # Set JAX settings
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    
    if args.mode == "demo":
        demo_code_generation()
    elif args.mode == "interactive":
        interactive_mode()


if __name__ == "__main__":
    main()