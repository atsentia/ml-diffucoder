# GRPO Training Guide for JAX DiffuCoder

This guide covers how to train DiffuCoder models using Coupled Group Relative Policy Optimization (GRPO) in JAX.

## Overview

Coupled GRPO is a policy optimization method specifically designed for diffusion language models. It combines:

- **Group Relative Policy Optimization**: Trains the model to generate higher-reward completions relative to a group baseline
- **Diffusion-Based Generation**: Uses masked token prediction with iterative denoising
- **Multiple Reward Functions**: Supports combining multiple reward signals for comprehensive code quality assessment

## Key Concepts

### 1. Reward Functions

Reward functions evaluate the quality of generated code completions. Common types include:

- **Syntax Correctness**: Checks if code compiles/parses correctly
- **Functional Correctness**: Validates against test cases
- **Code Quality**: Assesses formatting, style, documentation
- **Performance**: Measures efficiency and complexity

### 2. GRPO Algorithm

1. Generate multiple completions for each prompt
2. Compute rewards for all completions
3. Calculate advantages using group-wise baselines
4. Update policy to increase probability of high-advantage completions
5. Apply KL regularization to prevent over-optimization

### 3. Diffusion Process

The model uses masked token prediction with:
- Random masking ratios (0.2-0.8 by default)
- Multiple masked versions for robust training
- Weighted probability combinations

## Training Configuration

### Basic Configuration

```python
from jax_lm.training.config import TrainingConfig

config = TrainingConfig(
    # Model settings
    model_name_or_path="apple/DiffuCoder-7B-Base",
    dtype="bfloat16",
    
    # Training hyperparameters
    learning_rate=1e-6,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_grad_norm=0.2,
    
    # GRPO specific settings
    num_generations=4,          # Completions per prompt
    num_iterations=2,           # GRPO training iterations
    beta=0.01,                  # KL penalty coefficient
    epsilon_low=0.2,            # PPO clipping (lower bound)
    epsilon_high=0.2,           # PPO clipping (upper bound)
    scale_rewards=True,         # Normalize rewards by std dev
    
    # Diffusion settings
    diffusion_steps=64,         # Number of diffusion steps
    random_masking=True,        # Use random mask patterns
    generation_temperature=0.8, # Sampling temperature
    max_completion_length=256,  # Max tokens to generate
    
    # Reward settings
    reward_funcs=["syntax", "format", "tests"],
    reward_weights=[1.0, 0.3, 2.0],
    
    # System settings
    output_dir="./grpo_checkpoints",
    seed=42,
)
```

### Advanced Configuration

```python
config = TrainingConfig(
    # ... basic settings ...
    
    # Advanced GRPO settings
    sync_ref_model=True,        # Periodically sync reference model
    ref_model_sync_steps=64,    # Steps between ref model syncs
    
    # Advanced diffusion settings
    p_mask_prompt=0.1,          # Probability of masking prompt tokens
    generation_batch_size=4,    # Batch size for generation
    
    # Distributed training
    model_parallel_size=2,      # Model parallelism
    data_parallel_size=4,       # Data parallelism
    
    # Logging and checkpointing
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
    save_total_limit=5,
    
    # Wandb integration
    report_to=["wandb"],
    wandb_project="diffucoder-grpo",
    wandb_entity="your-org",
)
```

## Implementing Reward Functions

### 1. Syntax Correctness Reward

```python
import ast
import subprocess

class SyntaxReward:
    def __init__(self, language="python"):
        self.language = language
    
    def __call__(self, prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            try:
                if self.language == "python":
                    ast.parse(completion)
                    reward = 1.0
                else:
                    # Use appropriate parser for other languages
                    reward = self._check_syntax(completion)
            except SyntaxError:
                reward = -1.0
            except Exception:
                reward = 0.0
            rewards.append(reward)
        return rewards
    
    def _check_syntax(self, code):
        # Implement language-specific syntax checking
        pass
```

### 2. Test-Based Reward

```python
import tempfile
import subprocess
import json

class TestExecutionReward:
    def __init__(self, timeout=5.0):
        self.timeout = timeout
    
    def __call__(self, prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Extract test cases from prompt metadata
            tests = self._extract_tests(prompt, kwargs)
            reward = self._run_tests(completion, tests)
            rewards.append(reward)
        return rewards
    
    def _extract_tests(self, prompt, kwargs):
        # Extract test cases from problem description
        return kwargs.get("test_cases", [])
    
    def _run_tests(self, code, tests):
        if not tests:
            return 0.0
        
        passed = 0
        for test in tests:
            try:
                # Create temporary file with code + test
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code + "\n" + test)
                    temp_path = f.name
                
                # Run test with timeout
                result = subprocess.run(
                    ["python", temp_path],
                    timeout=self.timeout,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    passed += 1
                    
            except subprocess.TimeoutExpired:
                # Timeout penalty
                pass
            except Exception:
                # Other errors
                pass
            finally:
                # Clean up
                try:
                    import os
                    os.unlink(temp_path)
                except:
                    pass
        
        return passed / len(tests) if tests else 0.0
```

### 3. Code Quality Reward

```python
import re

class CodeQualityReward:
    def __init__(self):
        self.quality_checks = [
            ("has_docstring", self._has_docstring, 0.3),
            ("proper_naming", self._proper_naming, 0.2),
            ("good_comments", self._good_comments, 0.2),
            ("proper_structure", self._proper_structure, 0.3),
        ]
    
    def __call__(self, prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            reward = 0.0
            for name, check_fn, weight in self.quality_checks:
                if check_fn(completion):
                    reward += weight
            rewards.append(reward)
        return rewards
    
    def _has_docstring(self, code):
        return '"""' in code or "'''" in code
    
    def _proper_naming(self, code):
        # Check for snake_case functions, PascalCase classes
        functions = re.findall(r'def\s+(\w+)', code)
        classes = re.findall(r'class\s+(\w+)', code)
        
        proper_functions = all(name.islower() and '_' in name or name.islower() 
                              for name in functions)
        proper_classes = all(name[0].isupper() for name in classes)
        
        return proper_functions and proper_classes
    
    def _good_comments(self, code):
        # Check for meaningful comments
        comments = re.findall(r'#\s*(.+)', code)
        return len(comments) > 0 and any(len(comment.strip()) > 10 for comment in comments)
    
    def _proper_structure(self, code):
        # Check for proper indentation and structure
        lines = code.split('\n')
        proper_indent = all(
            line.startswith('    ') or line.startswith('\t') or line.strip() == '' or not line.startswith(' ')
            for line in lines if line.strip()
        )
        return proper_indent
```

## Training Process

### 1. Prepare Dataset

```python
from datasets import Dataset

def create_training_dataset():
    # Load your code generation dataset
    data = [
        {
            "prompt": "def fibonacci(n):",
            "test_cases": ["assert fibonacci(5) == 5", "assert fibonacci(0) == 0"],
            "difficulty": "easy",
        },
        # ... more examples
    ]
    
    return Dataset.from_list(data)

train_dataset = create_training_dataset()
eval_dataset = create_training_dataset()  # Use separate eval data
```

### 2. Initialize Trainer

```python
from jax_lm.training.coupled_grpo import DiffuGRPOTrainer
from jax_lm.utils.tokenizer import load_tokenizer

# Load tokenizer
tokenizer = load_tokenizer("apple/DiffuCoder-7B-Base")

# Setup reward functions
reward_functions = [
    SyntaxReward(),
    TestExecutionReward(),
    CodeQualityReward(),
]

# Initialize trainer
trainer = DiffuGRPOTrainer(
    config=config,
    reward_funcs=reward_functions,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
```

### 3. Start Training

```python
# Train the model
trainer.train()

# Model will be saved to config.output_dir
print(f"Training completed! Model saved to {config.output_dir}")
```

## Training Tips

### 1. Hyperparameter Tuning

- **Learning Rate**: Start with 1e-6, adjust based on loss curves
- **Beta (KL penalty)**: Start with 0.01, increase if model diverges too much
- **Num Generations**: More generations = better baselines but slower training
- **Epsilon (clipping)**: Standard PPO values (0.1-0.3) work well

### 2. Reward Function Design

- **Balance**: Combine multiple reward types for comprehensive evaluation
- **Scaling**: Ensure reward scales are comparable across functions
- **Stability**: Avoid extremely sparse or noisy rewards
- **Efficiency**: Optimize reward computation for speed

### 3. Memory Management

- Use `dtype=jnp.bfloat16` to reduce memory usage
- Adjust `generation_batch_size` based on available memory
- Enable gradient checkpointing for large models
- Use model parallelism for models that don't fit on single device

### 4. Monitoring Training

```python
# Key metrics to monitor:
# - Loss: Should decrease over time
# - Reward: Should increase over time
# - KL divergence: Should remain bounded
# - Clip ratios: Should be reasonable (not too high)

# Use Wandb for detailed tracking
import wandb
wandb.init(project="diffucoder-grpo")
```

## Common Issues and Solutions

### 1. Training Instability

**Symptoms**: Loss explodes, rewards become extremely high/low
**Solutions**:
- Reduce learning rate
- Increase KL penalty (beta)
- Check reward function bounds
- Enable gradient clipping

### 2. Slow Convergence

**Symptoms**: Rewards plateau quickly, little improvement
**Solutions**:
- Increase number of generations
- Improve reward function design
- Check dataset quality and diversity
- Adjust clipping parameters

### 3. Memory Issues

**Symptoms**: OOM errors during training
**Solutions**:
- Reduce batch sizes
- Use bfloat16 dtype
- Enable gradient checkpointing
- Use model parallelism

### 4. Poor Code Quality

**Symptoms**: Generated code has syntax errors, poor formatting
**Solutions**:
- Strengthen syntax reward function
- Add code quality rewards
- Increase training duration
- Check base model quality

## Advanced Topics

### 1. Custom Diffusion Strategies

```python
# Implement custom masking strategies
def custom_forward_process(batch, prompt_index, mask_id, key):
    # Your custom masking logic
    pass
```

### 2. Multi-Language Training

```python
# Support multiple programming languages
class MultiLanguageReward:
    def __init__(self):
        self.language_rewards = {
            "python": PythonReward(),
            "java": JavaReward(),
            "cpp": CppReward(),
        }
    
    def __call__(self, prompts, completions, **kwargs):
        languages = kwargs.get("languages", ["python"] * len(prompts))
        rewards = []
        for prompt, completion, lang in zip(prompts, completions, languages):
            reward_fn = self.language_rewards.get(lang, self.language_rewards["python"])
            reward = reward_fn([prompt], [completion])[0]
            rewards.append(reward)
        return rewards
```

### 3. Curriculum Learning

```python
# Gradually increase difficulty during training
class CurriculumDataset:
    def __init__(self, base_dataset, difficulty_schedule):
        self.base_dataset = base_dataset
        self.difficulty_schedule = difficulty_schedule
        self.current_step = 0
    
    def get_batch(self, step):
        max_difficulty = self.difficulty_schedule(step)
        # Filter dataset by difficulty
        return filtered_dataset
```

This guide should provide a comprehensive foundation for training DiffuCoder models with GRPO in JAX. Adjust the configuration and reward functions based on your specific use case and requirements.