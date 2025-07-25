# Training JAX DiffuCoder

This guide covers training DiffuCoder models using JAX/Flax, including the Coupled-GRPO algorithm.

## Overview

JAX DiffuCoder supports several training approaches:
1. **Supervised Fine-tuning (SFT)**: Standard supervised learning
2. **Coupled-GRPO**: Advanced reinforcement learning for code generation
3. **Continued Pre-training**: Further pre-training on domain-specific data

## Quick Start

### Basic Fine-tuning

```python
from jax_lm.training import Trainer, TrainingConfig

# Configure training
config = TrainingConfig(
    model_name_or_path="./models/jax",
    dataset_name="code_instructions.jsonl",
    learning_rate=1e-6,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    output_dir="./outputs",
)

# Create trainer
trainer = Trainer(config=config)

# Start training
trainer.train()
```

## Coupled-GRPO Training

Coupled-GRPO (Group Relative Policy Optimization) is designed specifically for diffusion models:

### How It Works

1. **Coupled Sampling**: For each example, sample complementary mask pairs
2. **Reward Computation**: Evaluate code quality using multiple metrics
3. **Policy Update**: Update model to increase probability of high-reward outputs

### Implementation

```python
from jax_lm.training import CoupledGRPOTrainer, TrainingConfig

config = TrainingConfig(
    model_name_or_path="apple/DiffuCoder-7B-Base",
    dataset_name="acecode_hard.jsonl",
    
    # GRPO specific
    beta=0.01,  # KL penalty
    epsilon=0.5,  # Clipping parameter
    num_generations=10,  # Samples per prompt
    num_iterations=2,  # GRPO iterations
    
    # Rewards
    reward_funcs=["code", "code_format"],
    reward_weights=[2.0, 0.5],
    e2b_router_url="localhost:8000",  # Code execution service
    
    # Training
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
)

trainer = CoupledGRPOTrainer(config)
trainer.train()
```

## Data Preparation

### Format Requirements

Training data should be in JSONL format:

```json
{"question": "Write a function to calculate factorial", "answer": "def factorial(n):\n    ..."}
{"question": "Implement binary search", "answer": "def binary_search(arr, target):\n    ..."}
```

### Processing Script

```bash
python recipes/process_data.py \
    --dataset_path "TIGER-Lab/AceCode-89K" \
    --output_path "./data/training.jsonl" \
    --difficulty "hard"
```

## Training Configuration

### Key Parameters

```python
@dataclass
class TrainingConfig:
    # Model
    model_name_or_path: str
    dtype: str = "bfloat16"  # or "float32", "float16"
    
    # Data
    dataset_name: str
    max_prompt_length: int = 512
    max_completion_length: int = 512
    
    # Optimization
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    
    # Schedule
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    report_to: List[str] = ["wandb"]
```

### Hardware Configuration

#### Single GPU
```python
config = TrainingConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 16
)
```

#### Multi-GPU (Data Parallel)
```python
# Automatically detected and used
# Effective batch size = per_device_batch * num_devices * grad_accum
```

#### TPU Training
```python
# Set TPU-specific flags
config = TrainingConfig(
    tpu_num_cores=8,
    per_device_train_batch_size=16,  # TPUs handle larger batches
    dtype="bfloat16",  # Native TPU precision
)
```

## Custom Training Loop

For advanced use cases, implement a custom training loop:

```python
import jax
import optax
from flax.training import train_state

# Create optimizer
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=1e-6, weight_decay=0.1)
)

# Initialize training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
)

# Training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch["input_ids"])
        loss = compute_cross_entropy(logits, batch["labels"])
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        state, loss = train_step(state, batch)
```

## Distributed Training

### Data Parallelism

```python
from jax.experimental import multihost_utils

# Initialize distributed training
jax.distributed.initialize()

# Shard data across devices
sharded_batch = multihost_utils.host_local_array_to_global_array(
    batch, mesh, P("data")
)
```

### Model Parallelism

```python
from jax.sharding import PartitionSpec as P

# Define sharding strategy
def get_sharding_spec(params):
    return jax.tree_map(
        lambda x: P("model", "data") if x.ndim > 1 else P("data"),
        params
    )

# Apply sharding
sharded_params = jax.device_put(params, sharding_spec)
```

## Monitoring Training

### Weights & Biases Integration

```python
import wandb

wandb.init(
    project="diffucoder-training",
    config=config.__dict__,
)

# Log metrics
wandb.log({
    "loss": loss,
    "learning_rate": current_lr,
    "tokens_per_second": tps,
})
```

### TensorBoard

```python
from flax.metrics import tensorboard

summary_writer = tensorboard.SummaryWriter("./logs")
summary_writer.scalar("loss", loss, step)
```

## Fine-tuning Best Practices

### 1. Learning Rate Selection
- Start with 1e-6 for fine-tuning
- Use learning rate warmup (10% of steps)
- Consider layer-wise learning rates

### 2. Batch Size Optimization
- Larger batches generally better for stability
- Use gradient accumulation if memory limited
- Monitor gradient norms

### 3. Regularization
- Weight decay: 0.1 for most cases
- Dropout: Usually not needed for fine-tuning
- Gradient clipping: Essential (max_norm=1.0)

### 4. Early Stopping
```python
# Monitor validation loss
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter > patience:
        print("Early stopping triggered")
        break
```

## Troubleshooting

### Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing:
```python
config.gradient_checkpointing = True
```

3. Use mixed precision:
```python
config.dtype = "bfloat16"
```

4. Implement gradient accumulation

### Slow Training

1. Ensure JIT compilation:
```python
train_step = jax.jit(train_step_fn)
```

2. Profile performance:
```python
with jax.profiler.trace("/tmp/jax-trace"):
    state, loss = train_step(state, batch)
```

3. Check data loading bottlenecks

### Loss Explosion

1. Reduce learning rate
2. Increase gradient clipping
3. Check for numerical instabilities
4. Use loss scaling for float16

## Example Training Script

Complete example for fine-tuning:

```python
#!/usr/bin/env python3
import jax
from jax_lm.training import CoupledGRPOTrainer, TrainingConfig
from datasets import load_dataset

def main():
    # Load dataset
    dataset = load_dataset("json", data_files="code_instructions.jsonl")
    
    # Configure training
    config = TrainingConfig(
        model_name_or_path="apple/DiffuCoder-7B-Base",
        dataset_name="code_instructions",
        
        # Hyperparameters
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_ratio=0.1,
        
        # Output
        output_dir="./outputs/diffucoder-finetuned",
        logging_dir="./logs",
        
        # Evaluation
        eval_steps=500,
        save_steps=1000,
        
        # Hardware
        dtype="bfloat16",
    )
    
    # Create trainer
    trainer = CoupledGRPOTrainer(
        config=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model("./final_model")
    
if __name__ == "__main__":
    main()
```

## Next Steps

1. **Experiment with hyperparameters**: Learning rate, batch size, warmup
2. **Try different reward functions**: Code execution, syntax checking, style
3. **Implement custom metrics**: BLEU, CodeBLEU, exact match
4. **Scale to larger models**: Use model parallelism for 13B+ parameters