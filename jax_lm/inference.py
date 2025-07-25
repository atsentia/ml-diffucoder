#!/usr/bin/env python3
"""Simplified JAX DiffuCoder inference script.

This script provides a simple interface for running inference with DiffuCoder
on JAX, optimized for TPU/GPU/CPU.
"""

import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Add parent directory to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_lm.models.dream import DreamForCausalLM, DreamConfig
from jax_lm.generate_diffusion import diffusion_generate
from jax_lm.utils.tokenizer import load_tokenizer, prepare_input_ids, decode_sequences


class DiffuCoderInference:
    """Simple inference wrapper for DiffuCoder."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        dtype: jnp.dtype = jnp.bfloat16,
        device: Optional[str] = None,
    ):
        """Initialize DiffuCoder for inference.
        
        Args:
            model_path: Path to JAX model directory with config.json and params.pkl
            tokenizer_path: Path to tokenizer (if None, tries model_path)
            dtype: Model dtype (bfloat16 recommended for TPU)
            device: Device to use (if None, auto-detects)
        """
        self.model_path = Path(model_path)
        self.dtype = dtype
        
        # Device setup
        self._setup_device(device)
        
        # Load model
        self.model, self.params = self._load_model()
        
        # Load tokenizer
        if tokenizer_path is None:
            # Try common locations
            tokenizer_candidates = [
                self.model_path,
                self.model_path.parent / "diffucoder-7b-complete",
                Path("./models/diffucoder-7b-complete"),
            ]
            for candidate in tokenizer_candidates:
                if candidate.exists():
                    tokenizer_path = candidate
                    break
        
        if tokenizer_path is None:
            raise ValueError("Could not find tokenizer. Please specify tokenizer_path.")
        
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Get special tokens
        self.mask_token_id = self.model.config.mask_token_id
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device_type}")
        print(f"   Parameters: {self._count_params()/1e9:.1f}B")
        print(f"   Dtype: {self.dtype}")
    
    def _setup_device(self, device: Optional[str]):
        """Setup JAX device."""
        available_devices = jax.devices()
        print(f"Available devices: {available_devices}")
        
        if device:
            # User specified device
            self.device_type = device
        else:
            # Auto-detect
            device_types = [str(d.platform).lower() for d in available_devices]
            if "tpu" in device_types:
                self.device_type = "tpu"
            elif "gpu" in device_types or "cuda" in device_types:
                self.device_type = "gpu"
            else:
                self.device_type = "cpu"
        
        print(f"Using device: {self.device_type}")
    
    def _load_model(self) -> tuple:
        """Load model and weights."""
        # Load config
        config_path = self.model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        
        # Create config with proper mappings
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
            attention_dropout=config_dict.get("attention_dropout", 0.0),
            # Special tokens
            mask_token_id=config_dict.get("mask_token_id", 151666),
            pad_token_id=config_dict.get("pad_token_id", 151643),
            bos_token_id=config_dict.get("bos_token_id", 151643),
            eos_token_id=config_dict.get("eos_token_id", 151643),
            dtype=self.dtype,
        )
        
        # Create model
        model = DreamForCausalLM(config=config, dtype=self.dtype)
        
        # Load weights
        params_path = self.model_path / "params.pkl"
        print(f"Loading weights from {params_path}...")
        
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        
        # Convert to specified dtype if needed
        if self.dtype != jnp.float32:
            params = jax.tree_map(lambda x: x.astype(self.dtype), params)
        
        return model, params
    
    def _count_params(self) -> int:
        """Count total parameters."""
        return sum(x.size for x in jax.tree_util.tree_leaves(self.params))
    
    @jax.jit
    def _forward(self, params, input_ids, attention_mask=None):
        """JIT-compiled forward pass."""
        return self.model.apply(
            params,
            input_ids,
            attention_mask=attention_mask,
            deterministic=True,
        )
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.95,
        num_steps: Optional[int] = None,
        tokens_per_step: int = 1,
        seed: int = 42,
        return_dict: bool = False,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """Generate text from prompt(s).
        
        Args:
            prompt: Input prompt(s)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            num_steps: Number of diffusion steps (if None, defaults to max_new_tokens)
            tokens_per_step: Tokens to unmask per step
            seed: Random seed
            return_dict: If True, return full output dictionary
            
        Returns:
            Generated text(s) or full output dictionary
        """
        # Handle single vs batch
        is_single = isinstance(prompt, str)
        if is_single:
            prompt = [prompt]
        
        # Tokenize
        inputs = prepare_input_ids(
            self.tokenizer,
            prompt,
            padding=True,
            truncation=True,
            return_tensors="jax",
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # Set num_steps if not provided
        if num_steps is None:
            num_steps = max_new_tokens
        
        # Generate
        print(f"Generating {max_new_tokens} tokens...")
        start_time = time.time()
        
        # Create RNG
        rng = random.PRNGKey(seed)
        
        # Run generation
        outputs = diffusion_generate(
            self.model,
            self.params,
            input_ids,
            rng,
            attention_mask=attention_mask,
            num_steps=num_steps,
            tokens_per_step=tokens_per_step,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            alg="entropy",
        )
        
        # Wait for completion
        outputs["sequences"].block_until_ready()
        generation_time = time.time() - start_time
        
        # Extract generated portion
        prompt_lengths = jnp.sum(attention_mask, axis=1)
        generated_sequences = outputs["sequences"]
        
        # Decode
        generated_texts = []
        for i, seq in enumerate(generated_sequences):
            # Get only the generated part
            prompt_len = int(prompt_lengths[i])
            generated_tokens = seq[prompt_len:prompt_len + max_new_tokens]
            
            # Remove mask tokens and padding
            valid_tokens = generated_tokens[
                (generated_tokens != self.mask_token_id) & 
                (generated_tokens != self.tokenizer.pad_token_id)
            ]
            
            # Decode
            text = decode_sequences(self.tokenizer, valid_tokens)
            generated_texts.append(text)
        
        print(f"Generation completed in {generation_time:.2f}s")
        print(f"Tokens/second: {max_new_tokens * len(prompt) / generation_time:.2f}")
        
        if return_dict:
            return {
                "generated_texts": generated_texts[0] if is_single else generated_texts,
                "sequences": outputs["sequences"],
                "generation_time": generation_time,
                "tokens_per_second": max_new_tokens * len(prompt) / generation_time,
            }
        else:
            return generated_texts[0] if is_single else generated_texts
    
    def benchmark_inference(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        seq_length: int = 128,
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark inference performance.
        
        Args:
            batch_sizes: List of batch sizes to test
            seq_length: Sequence length for benchmarking
            num_iterations: Number of iterations per batch size
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking batch size {batch_size}...")
            
            # Create dummy input
            dummy_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
            
            # Warmup
            _ = self._forward(self.params, dummy_ids)
            
            # Time iterations
            times = []
            for _ in range(num_iterations):
                start = time.time()
                output = self._forward(self.params, dummy_ids)
                output["logits"].block_until_ready()
                times.append(time.time() - start)
            
            avg_time = np.mean(times[2:])  # Skip first two for stability
            throughput = batch_size * seq_length / avg_time
            
            results[f"batch_{batch_size}"] = {
                "avg_time_seconds": avg_time,
                "throughput_tokens_per_second": throughput,
            }
            
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Throughput: {throughput:.0f} tokens/s")
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="DiffuCoder JAX inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/dream-jax",
        help="Path to JAX model directory",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (if different from model path)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="def fibonacci(n):",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling threshold",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark instead of generation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype",
    )
    
    args = parser.parse_args()
    
    # Map dtype string to JAX dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize model
    model = DiffuCoderInference(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dtype=dtype,
    )
    
    if args.benchmark:
        # Run benchmark
        results = model.benchmark_inference()
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))
    else:
        # Generate text
        output = model.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {output}")


if __name__ == "__main__":
    main()