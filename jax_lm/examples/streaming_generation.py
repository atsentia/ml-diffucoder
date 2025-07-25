#!/usr/bin/env python3
"""Streaming generation example for JAX DiffuCoder."""

import time
import jax
import jax.numpy as jnp
from jax_lm import load_model
from jax_lm.utils import load_tokenizer
from jax_lm.generate_diffusion import diffusion_generate


class StreamingGenerator:
    """Helper class for streaming generation with callbacks."""
    
    def __init__(self, model, params, tokenizer):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.generated_tokens = []
        self.mask_history = []
    
    def stream_callback(self, state):
        """Callback function called during generation."""
        step = state.get("step", 0)
        sequences = state.get("sequences")
        mask = state.get("mask")
        
        if sequences is not None:
            # Count unmasked tokens
            unmasked = jnp.sum(mask == 0)
            total = mask.size
            progress = (total - unmasked) / total * 100
            
            # Decode current state
            current_text = self.tokenizer.decode(
                sequences[0], 
                skip_special_tokens=False
            )
            
            # Print progress
            print(f"\rStep {step}: {progress:.1f}% complete ({unmasked}/{total} tokens)", 
                  end="", flush=True)
            
            # Store history
            self.mask_history.append(mask.copy())
    
    def generate_with_streaming(self, prompt, rng, **kwargs):
        """Generate with streaming updates."""
        print(f"Prompt: {prompt}")
        print("Generating...\n")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="jax")
        input_ids = inputs["input_ids"]
        
        # Custom generation loop with callbacks
        sequences = jnp.ones((1, len(prompt) + kwargs.get("max_new_tokens", 256)), 
                            dtype=jnp.int32) * self.model.config.mask_token_id
        sequences = sequences.at[:, :input_ids.shape[1]].set(input_ids)
        
        mask = jnp.ones_like(sequences, dtype=jnp.float32)
        mask = mask.at[:, :input_ids.shape[1]].set(0)
        
        num_steps = kwargs.get("num_steps", 256)
        tokens_per_step = kwargs.get("tokens_per_step", 1)
        
        start_time = time.time()
        
        for step in range(num_steps):
            # Forward pass
            outputs = self.model.apply(
                self.params,
                sequences,
                deterministic=True,
            )
            logits = outputs["logits"]
            
            # Sample tokens
            rng, sample_rng = jax.random.split(rng)
            temperature = kwargs.get("temperature", 0.3)
            
            # Simple sampling for masked positions
            probs = jax.nn.softmax(logits / temperature, axis=-1)
            sampled = jax.random.categorical(sample_rng, jnp.log(probs), axis=-1)
            
            # Update only masked positions
            sequences = jnp.where(mask.astype(bool), sampled, sequences)
            
            # Determine positions to unmask (simplified)
            if jnp.sum(mask) > 0:
                # Unmask random positions for demo
                rng, unmask_rng = jax.random.split(rng)
                masked_positions = jnp.where(mask.reshape(-1) == 1)[0]
                
                if len(masked_positions) > 0:
                    num_unmask = min(tokens_per_step, len(masked_positions))
                    positions = jax.random.choice(
                        unmask_rng,
                        masked_positions,
                        shape=(num_unmask,),
                        replace=False
                    )
                    
                    for pos in positions:
                        mask = mask.at[0, pos].set(0)
            
            # Callback
            self.stream_callback({
                "step": step,
                "sequences": sequences,
                "mask": mask,
            })
            
            # Check if done
            if jnp.sum(mask) == 0:
                break
        
        elapsed = time.time() - start_time
        print(f"\n\nGeneration complete in {elapsed:.2f}s")
        
        # Final result
        generated_text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        return generated_text, {
            "sequences": sequences,
            "steps": step + 1,
            "time": elapsed,
        }


def main():
    # Setup
    model_path = "./models/jax"
    
    print("Loading model and tokenizer...")
    model, params = load_model(model_path, dtype=jnp.bfloat16)
    tokenizer = load_tokenizer(f"{model_path}/tokenizer")
    
    # Create streaming generator
    generator = StreamingGenerator(model, params, tokenizer)
    
    # Example prompts
    prompts = [
        "def bubble_sort(arr):",
        "class Calculator:\n    def __init__(self):",
    ]
    
    rng = jax.random.PRNGKey(42)
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        
        rng, gen_rng = jax.random.split(rng)
        generated_text, stats = generator.generate_with_streaming(
            prompt,
            gen_rng,
            max_new_tokens=128,
            temperature=0.3,
            num_steps=128,
            tokens_per_step=2,
        )
        
        print(f"\nFinal output:")
        print(generated_text)
        
        print(f"\nStatistics:")
        print(f"  Steps: {stats['steps']}")
        print(f"  Time: {stats['time']:.2f}s")
        print(f"  Tokens/second: {128 / stats['time']:.1f}")


if __name__ == "__main__":
    main()