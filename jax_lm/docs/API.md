# JAX DiffuCoder API Reference

## Core Functions

### Model Loading and Saving

#### `load_model(model_path, dtype=jnp.float32, device=None)`
Load a DiffuCoder model from disk.

**Parameters:**
- `model_path` (str): Path to model directory or HuggingFace model ID
- `dtype` (dtype): Data type for model parameters (default: float32)
- `device` (str, optional): Device to load model on

**Returns:**
- `(model, params)`: Model instance and parameters

**Example:**
```python
model, params = load_model("./models/jax", dtype=jnp.bfloat16)
```

#### `save_model(model, params, save_path)`
Save a DiffuCoder model to disk.

**Parameters:**
- `model`: DiffuCoder model instance
- `params`: Model parameters
- `save_path` (str): Directory to save model

#### `convert_pytorch_checkpoint(pytorch_path, output_path, dtype=jnp.float32)`
Convert PyTorch checkpoint to JAX format.

**Parameters:**
- `pytorch_path` (str): Path to PyTorch checkpoint
- `output_path` (str): Path to save JAX checkpoint
- `dtype` (dtype): Target data type

**Returns:**
- `(model, params)`: Converted model and parameters

### Generation Functions

#### `diffusion_generate(model, params, input_ids, rng, **kwargs)`
Generate text using masked diffusion process.

**Parameters:**
- `model`: DiffuCoder model instance
- `params`: Model parameters
- `input_ids` (array): Input token IDs [batch_size, seq_len]
- `rng`: JAX random key
- `attention_mask` (array, optional): Attention mask
- `num_steps` (int): Number of diffusion steps (default: 256)
- `tokens_per_step` (int): Tokens to unmask per step (default: 1)
- `max_new_tokens` (int): Maximum tokens to generate (default: 256)
- `temperature` (float): Sampling temperature (default: 0.3)
- `top_p` (float): Nucleus sampling threshold (default: 0.95)
- `alg` (str): Unmasking algorithm ("entropy" or "random")
- `alg_temp` (float): Temperature for algorithm

**Returns:**
- Dictionary with:
  - `sequences`: Generated sequences
  - `scores`: Generation scores
  - `history`: Generation history
  - `attention_mask`: Final attention mask

**Example:**
```python
output = diffusion_generate(
    model, params, input_ids, rng,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.95,
    alg="entropy"
)
```

#### `stream_diffusion_generate(model, params, input_ids, rng, callback, **kwargs)`
Streaming version of diffusion generation.

**Parameters:**
- Same as `diffusion_generate`
- `callback` (callable): Function called after each step

**Example:**
```python
def print_progress(state):
    print(f"Step {state['step']}: {state['unmasked_tokens']} tokens")

stream_diffusion_generate(
    model, params, input_ids, rng,
    callback=print_progress
)
```

## Model Classes

### `DiffuCoder`
Main model class implementing masked diffusion transformer.

**Methods:**
- `__call__(input_ids, attention_mask=None, position_ids=None, deterministic=True)`
  - Forward pass through the model
  - Returns logits or dictionary with outputs

**Example:**
```python
model = DiffuCoder(config)
outputs = model.apply(params, input_ids, deterministic=True)
logits = outputs["logits"]
```

### `DiffuCoderConfig`
Configuration class for DiffuCoder models.

**Attributes:**
- `vocab_size` (int): Vocabulary size (default: 32000)
- `hidden_size` (int): Hidden dimension size (default: 4096)
- `num_hidden_layers` (int): Number of transformer layers (default: 32)
- `num_attention_heads` (int): Number of attention heads (default: 32)
- `num_key_value_heads` (int, optional): For grouped-query attention
- `intermediate_size` (int): FFN intermediate size (default: 11008)
- `hidden_act` (str): Activation function (default: "silu")
- `max_position_embeddings` (int): Maximum sequence length (default: 4096)
- `rope_theta` (float): RoPE theta parameter (default: 10000.0)
- `diffusion_steps` (int): Default diffusion steps (default: 256)
- `mask_token_id` (int): Mask token ID (default: 32001)
- `pad_token_id` (int): Padding token ID (default: 32002)

**Example:**
```python
config = DiffuCoderConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=24
)
```

## Training Classes

### `TrainingConfig`
Configuration for training DiffuCoder models.

**Key Attributes:**
- `model_name_or_path` (str): Model path or name
- `learning_rate` (float): Learning rate
- `num_train_epochs` (int): Number of epochs
- `per_device_train_batch_size` (int): Batch size per device
- `gradient_accumulation_steps` (int): Gradient accumulation
- `warmup_ratio` (float): Warmup ratio
- `weight_decay` (float): Weight decay
- `save_steps` (int): Save checkpoint frequency
- `logging_steps` (int): Logging frequency

### `Trainer`
Base trainer class for DiffuCoder models.

**Methods:**
- `train()`: Start training
- `evaluate()`: Run evaluation
- `save_checkpoint(state, metrics=None)`: Save checkpoint

**Example:**
```python
trainer = Trainer(
    config=training_config,
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data
)
trainer.train()
```

### `CoupledGRPOTrainer`
Specialized trainer for Coupled-GRPO algorithm.

**Additional Parameters:**
- `beta` (float): KL penalty coefficient
- `num_generations` (int): Generations per iteration
- `reward_funcs` (list): Reward functions to use

## Utility Functions

### Tokenization

#### `load_tokenizer(tokenizer_path, trust_remote_code=True)`
Load tokenizer from path or HuggingFace.

**Returns:**
- Tokenizer instance with special tokens configured

#### `prepare_input_ids(tokenizer, texts, max_length=None, **kwargs)`
Prepare input IDs for model input.

**Parameters:**
- `tokenizer`: Tokenizer instance
- `texts` (str or list): Input text(s)
- `max_length` (int, optional): Maximum sequence length
- `padding` (bool): Whether to pad sequences
- `truncation` (bool): Whether to truncate sequences
- `return_tensors` (str): "np" or "jax"

### Hardware Utilities

#### `setup_tpu()`
Initialize TPU environment.

**Returns:**
- List of TPU devices

#### `get_tpu_mesh(devices=None)`
Create mesh for data parallelism.

**Parameters:**
- `devices` (list, optional): Device list

**Returns:**
- JAX Mesh object

#### `shard_params(params, mesh, model_parallel_size=1)`
Shard model parameters across devices.

**Parameters:**
- `params`: Model parameters
- `mesh`: JAX mesh object
- `model_parallel_size` (int): Model parallel dimension

**Returns:**
- Sharded parameters

### Model Utilities

#### `count_parameters(params)`
Count total parameters in model.

**Returns:**
- Total parameter count

#### `parameter_summary(params)`
Get parameter breakdown by layer type.

**Returns:**
- Dictionary with parameter counts by category

## Advanced Usage

### Custom Generation Algorithms

```python
def custom_unmask_fn(logits, mask, step):
    """Custom unmasking strategy."""
    # Your logic here
    return positions_to_unmask

output = diffusion_generate(
    model, params, input_ids, rng,
    unmask_fn=custom_unmask_fn
)
```

### Batch Processing

```python
# Efficient batch generation
batch_size = 8
prompts = ["prompt1", "prompt2", ..., "prompt8"]

# Tokenize batch
inputs = tokenizer(prompts, padding=True, return_tensors="jax")

# Generate batch
outputs = diffusion_generate(
    model, params, 
    inputs["input_ids"],
    rng,
    attention_mask=inputs["attention_mask"]
)
```

### Mixed Precision

```python
# Convert model to bfloat16
params_bf16 = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)

# Use in generation
output = diffusion_generate(model, params_bf16, input_ids, rng)
```

### Distributed Generation

```python
# Setup for multi-device generation
devices = jax.devices()
mesh = Mesh(devices, axis_names=("data",))

with mesh:
    sharded_params = shard_params(params, mesh)
    
    # Distributed generation
    output = diffusion_generate(
        model, sharded_params, input_ids, rng
    )
```