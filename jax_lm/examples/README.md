# JAX DiffuCoder Examples

This directory contains example scripts demonstrating various features of JAX DiffuCoder.

## Prerequisites

Before running these examples, ensure you have:

1. Installed JAX DiffuCoder:
   ```bash
   pip install -e jax_lm
   ```

2. Downloaded and converted model weights:
   ```bash
   # Download from HuggingFace
   python jax_lm/scripts/download_weights.py \
       --model-id apple/DiffuCoder-7B-Base \
       --output-dir ./models/pytorch

   # Convert to JAX format
   python jax_lm/scripts/convert_pytorch_to_jax.py \
       --pytorch-path ./models/pytorch \
       --output-path ./models/jax
   ```

## Examples

### 1. Basic Generation (`basic_generation.py`)

Simple code generation for various programming tasks:

```bash
python basic_generation.py
```

Features:
- Single prompt generation
- Various programming languages
- Generation statistics

### 2. Batch Generation (`batch_generation.py`)

Efficient generation for multiple prompts:

```bash
python batch_generation.py
```

Features:
- Batch processing for efficiency
- Multiple programming languages
- Performance metrics

### 3. Streaming Generation (`streaming_generation.py`)

Real-time generation with progress updates:

```bash
python streaming_generation.py
```

Features:
- Live progress tracking
- Token-by-token updates
- Generation visualization

### 4. Custom Unmasking (`custom_unmasking.py`)

Implement custom token unmasking strategies:

```bash
python custom_unmasking.py
```

Features:
- Custom unmasking algorithms
- Syntax-aware generation
- Language-specific strategies

### 5. Multi-Device Generation (`multi_device.py`)

Utilize multiple GPUs/TPUs:

```bash
python multi_device.py
```

Features:
- Automatic device detection
- Model sharding
- Distributed generation

### 6. GRPO Training (`grpo_training.py`)

Complete example of training DiffuCoder with Coupled Group Relative Policy Optimization (GRPO):

```bash
# Set up environment
export WANDB_ENTITY=your_wandb_entity
export JAX_PLATFORM_NAME=tpu  # or cpu for CPU training

# Run training
python grpo_training.py
```

Features:
- Reward function examples (syntax checking, code formatting)
- Custom dataset creation
- TPU/CPU support
- Wandb logging integration
- Checkpointing and resuming

### 7. GRPO Inference (`grpo_inference.py`)

Demonstrates inference with GRPO-trained models:

```bash
# Run demo with example prompts
python grpo_inference.py --mode demo --model-path ./grpo_checkpoints

# Interactive mode for custom prompts
python grpo_inference.py --mode interactive --model-path ./grpo_checkpoints
```

Features:
- Multiple completion generation
- Diffusion-based sampling
- Interactive prompt testing
- Configurable generation parameters

## Common Patterns

### Loading Models

```python
from jax_lm import load_model, load_tokenizer

model, params = load_model("./models/jax", dtype=jnp.bfloat16)
tokenizer = load_tokenizer("./models/jax/tokenizer")
```

### Basic Generation

```python
output = diffusion_generate(
    model, params, input_ids, rng,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.95,
)
```

### Handling Batches

```python
# Tokenize batch
inputs = tokenizer(prompts, padding=True, return_tensors="jax")

# Generate batch
outputs = diffusion_generate(
    model, params, 
    inputs["input_ids"],
    rng,
    attention_mask=inputs["attention_mask"],
)
```

### GRPO Training

```python
from jax_lm.training.coupled_grpo import DiffuGRPOTrainer
from jax_lm.training.config import TrainingConfig

# Setup configuration
config = TrainingConfig(
    model_name_or_path="apple/DiffuCoder-7B-Base",
    num_iterations=2,
    num_generations=4,
    beta=0.01,  # KL penalty
    reward_funcs=["code_syntax", "code_format"],
)

# Initialize trainer with reward functions
trainer = DiffuGRPOTrainer(
    config=config,
    reward_funcs=reward_functions,
    train_dataset=train_dataset,
)

# Start training
trainer.train()
```

### Custom Reward Functions for GRPO

```python
class CodeQualityReward:
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Implement your reward logic
            reward = evaluate_code_quality(completion)
            rewards.append(reward)
        return rewards
```

## Tips

1. **Memory Management**: Use `dtype=jnp.bfloat16` for reduced memory usage
2. **Performance**: JIT compile generation functions for better performance
3. **Debugging**: Set `JAX_TRACEBACK_FILTERING=off` for full error traces
4. **Device Selection**: Use `CUDA_VISIBLE_DEVICES` or `JAX_PLATFORM_NAME`

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use smaller model variant
- Enable CPU offloading

### Slow Generation
- Ensure JIT compilation
- Check device utilization
- Profile with JAX profiler

### Import Errors
- Verify installation: `pip show jax_lm`
- Check Python path
- Reinstall dependencies