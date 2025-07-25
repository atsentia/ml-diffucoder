#!/usr/bin/env python3
"""
Example script for training DiffuCoder with Coupled-GRPO in JAX.

This script demonstrates how to set up and run GRPO training for code generation
with diffusion language models.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the jax_lm package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import datasets
from datasets import Dataset

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.training.coupled_grpo import DiffuGRPOTrainer
from jax_lm.training.config import TrainingConfig
from jax_lm.utils.tokenizer import load_tokenizer


class CodeRewardFunction:
    """Simple code reward function based on syntax correctness."""
    
    def __init__(self):
        self.name = "code_syntax"
    
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Compute rewards based on code syntax correctness.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            
        Returns:
            List of reward scores (higher is better)
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            try:
                # Try to compile the completion as Python code
                compile(completion, '<string>', 'exec')
                # If successful, give positive reward
                reward = 1.0
            except SyntaxError:
                # If syntax error, give negative reward
                reward = -0.5
            except Exception:
                # Other exceptions get neutral reward
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards


class CodeFormatRewardFunction:
    """Reward function based on code formatting quality."""
    
    def __init__(self):
        self.name = "code_format"
    
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Compute rewards based on code formatting quality.
        
        Args:
            prompts: List of prompt strings  
            completions: List of completion strings
            
        Returns:
            List of reward scores (higher is better)
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            reward = 0.0
            
            # Check for proper indentation
            lines = completion.split('\n')
            if any(line.startswith('    ') or line.startswith('\t') for line in lines):
                reward += 0.2
            
            # Check for docstrings
            if '"""' in completion or "'''" in completion:
                reward += 0.2
            
            # Check for comments
            if '#' in completion:
                reward += 0.1
            
            # Check for proper function/class structure
            if 'def ' in completion or 'class ' in completion:
                reward += 0.3
            
            # Penalize very short completions
            if len(completion.strip()) < 10:
                reward -= 0.5
            
            rewards.append(reward)
        
        return rewards


def create_dummy_dataset(size: int = 100) -> Dataset:
    """Create a dummy dataset for demonstration purposes."""
    
    # Simple code generation prompts
    prompts = [
        "Write a function to calculate factorial of a number",
        "Create a class to represent a binary tree",
        "Implement quicksort algorithm",
        "Write a function to check if a string is palindrome",
        "Create a decorator for timing function execution",
        "Implement a simple calculator class",
        "Write a function to merge two sorted lists",
        "Create a context manager for file handling",
        "Implement breadth-first search for graphs",
        "Write a function to validate email addresses",
    ]
    
    # Cycle through prompts to create dataset of desired size
    dataset_prompts = [prompts[i % len(prompts)] for i in range(size)]
    
    dataset = Dataset.from_dict({
        "prompts": dataset_prompts,
        "task_type": ["code_generation"] * size,
        "difficulty": ["medium"] * size,
    })
    
    return dataset


def setup_training_config() -> TrainingConfig:
    """Setup training configuration for GRPO."""
    
    config = TrainingConfig(
        # Model settings
        model_name_or_path="apple/DiffuCoder-7B-Base",
        dtype="bfloat16",
        
        # Training settings
        learning_rate=1e-6,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_grad_norm=0.2,
        
        # GRPO specific settings
        num_generations=4,
        num_iterations=2,
        beta=0.01,
        epsilon_low=0.2,
        epsilon_high=0.2,
        scale_rewards=True,
        
        # Diffusion settings
        diffusion_steps=64,  # Reduced for faster training
        random_masking=True,
        generation_batch_size=2,
        generation_temperature=0.8,
        max_completion_length=256,
        
        # Reward settings
        reward_funcs=["code_syntax", "code_format"],
        reward_weights=[1.0, 0.5],
        
        # Logging and saving
        logging_steps=5,
        save_steps=50,
        eval_steps=25,
        output_dir="./grpo_checkpoints",
        overwrite_output_dir=True,
        
        # Wandb settings (optional)
        report_to=["wandb"],
        wandb_project="jax-diffucoder-grpo",
        wandb_entity=os.getenv("WANDB_ENTITY"),
        
        # System settings
        seed=42,
    )
    
    return config


def main():
    """Main training function."""
    
    print("🚀 Starting JAX DiffuCoder GRPO Training Example")
    print("=" * 60)
    
    # Setup configuration
    print("📋 Setting up training configuration...")
    config = setup_training_config()
    
    # Create datasets
    print("📚 Creating training dataset...")
    train_dataset = create_dummy_dataset(size=200)
    eval_dataset = create_dummy_dataset(size=50)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = load_tokenizer(config.model_name_or_path)
    
    # Setup reward functions
    print("🎯 Setting up reward functions...")
    reward_funcs = [
        CodeRewardFunction(),
        CodeFormatRewardFunction(),
    ]
    
    # Initialize trainer
    print("🏋️ Initializing GRPO trainer...")
    trainer = DiffuGRPOTrainer(
        config=config,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Print training info
    print("\n📊 Training Configuration:")
    print(f"  Model: {config.model_name_or_path}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  GRPO generations: {config.num_generations}")
    print(f"  GRPO iterations: {config.num_iterations}")
    print(f"  Diffusion steps: {config.diffusion_steps}")
    print(f"  Output directory: {config.output_dir}")
    
    # Check available devices
    print(f"\n🖥️  Available JAX devices: {jax.devices()}")
    print(f"  Device count: {jax.device_count()}")
    
    try:
        # Start training
        print("\n🚀 Starting GRPO training...")
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("💾 Saving current checkpoint...")
        trainer.save_checkpoint(trainer.state)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💾 Attempting to save current checkpoint...")
        try:
            trainer.save_checkpoint(trainer.state)
            print("✅ Checkpoint saved successfully")
        except:
            print("❌ Failed to save checkpoint")
        raise


if __name__ == "__main__":
    # Set environment variables for training
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # Change to "tpu" for TPU training
    
    main()