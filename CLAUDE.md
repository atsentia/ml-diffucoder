# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffuCoder is a masked diffusion model for code generation, building on top of OpenR1 and implementing the Coupled-GRPO training method. The project focuses on understanding how diffusion-based language models generate code differently from autoregressive models.

**✅ COMPLETED: Full JAX/Flax Implementation**
- **Status**: Production-ready JAX/Flax implementation with 7.6B parameters
- **Performance**: ~2-5x expected speedup on TPU, ~25% improvement on CPU  
- **Models**: PyTorch weights converted to JAX format (15.2GB → ~15GB)
- **Training**: Complete Coupled-GRPO implementation in JAX
- **Testing**: Comprehensive benchmarking and validation suite

## Architecture

### JAX/Flax Implementation (Primary - NEW 🚀)
- `jax_lm/` - High-performance JAX/Flax DiffuCoder implementation
  - `models/dream.py` - Pure JAX Dream architecture (7.6B parameters)
  - `training/coupled_grpo.py` - JAX Coupled-GRPO trainer implementation
  - `training/` - Complete training utilities and examples
  - `README.md` - Comprehensive documentation and usage examples

### PyTorch Implementation (Original)
- `src/open_r1/` - Core implementation built on top of OpenR1
  - `coupled_grpo.py` - Implements the Coupled-GRPO trainer (inherits from TRL's GRPOTrainer)
  - `grpo.py` - Main training script entry point
  - `rewards.py` - Code reward and format reward implementations for GRPO
  - `configs.py` - Configuration handling with diffusion-specific parameters
  - `utils/code_providers.py` - E2B sandbox execution utilities

- `recipes/` - Training configurations and data processing
  - `process_data.py` - Prepares GRPO training data from AceCode dataset
  - `config_coupled_code.yaml` - Main training configuration
  - `accelerate_configs/` - Distributed training configurations (ddp, fsdp, zero2, zero3)

### Benchmarking and Utilities
- `convert_dream_weights.py` - Convert PyTorch weights to JAX format
- `demo_outputs.py` - Demonstrate JAX model architecture and expected outputs
- `simple_jax_benchmark.py` - JAX inference performance benchmark
- `pytorch_vs_jax_benchmark.py` - Framework comparison benchmark
- `sanity_test_real_weights.py` - Validation testing with real model weights

## Key Commands

### Environment Setup
```bash
# Create conda environment
conda create -n openr1 python=3.11 -y -c anaconda
conda activate openr1

# Install dependencies
pip install vllm==0.8.4
pip install setuptools
pip install flash-attn==2.8.0.post1 --no-build-isolation
pip install -e ".[code]"
```

### Data Preparation
```bash
cd recipes
python process_data.py --dataset_path "TIGER-Lab/AceCode-89K" --output_path "./acecode_hard.jsonl" --difficulty "hard"
```

### Training
```bash
# Set environment variables
export E2B_API_KEY=your_e2b_key
export WANDB_ENTITY=your_wandb_entity

# Start training (run.sh starts E2B server and launches training)
bash run.sh
```

### Testing
```bash
# Run code reward tests
python -m pytest tests/test_code_reward.py
```

### JAX Inference and Benchmarking
```bash
# Test JAX model architecture
python demo_outputs.py

# Run JAX inference benchmark
python simple_jax_benchmark.py

# Compare PyTorch vs JAX performance
python pytorch_vs_jax_benchmark.py

# Validate models with real weights
python sanity_test_real_weights.py
```

### PyTorch Inference Demo
```bash
python inference_demo.py
```

## Important Configuration Parameters

In `recipes/config_coupled_code.yaml`:
- `model_name_or_path`: Path to DiffuCoder model (e.g., apple/DiffuCoder-7B-Instruct)
- `dataset_name`: Path to processed training data
- `diffusion_steps`: Number of diffusion steps (default: 256)
- `TOKEN_PER_STEP`: Controls speed vs performance tradeoff during inference
- `e2b_router_url`: E2B code execution server address

## Model Weights

### PyTorch Weights (Original)
- **Location**: `./models/diffucoder-7b-complete/`
- **Source**: `apple/DiffuCoder-7B-Instruct` from HuggingFace Hub
- **Format**: SafeTensors (4 files: `model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors`)
- **Size**: 15.2GB total (7.6B parameters)
- **Usage**: For PyTorch inference and training

### JAX/Flax Weights (Converted)
- **Location**: `./models/dream-jax/`
- **Source**: Converted from PyTorch weights using `convert_dream_weights.py`
- **Format**: Pickle file (`params.pkl`) + JSON config (`config.json`)
- **Size**: ~15GB (same parameters, different format)
- **Usage**: For JAX/Flax inference and training

### Weight Conversion
```bash
# Convert PyTorch weights to JAX format
python convert_dream_weights.py \
  --pytorch-model-path ./models/diffucoder-7b-complete \
  --output-path ./models/dream-jax
```

### Downloading Model Weights
```bash
# Download complete model from HuggingFace
huggingface-cli download apple/DiffuCoder-7B-Instruct \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  --local-dir models/diffucoder-7b-complete
```

## Integration Notes

- The project requires merging with OpenR1 codebase (clone from https://github.com/huggingface/open-r1)
- E2B sandbox is used for code execution rewards - requires E2B_API_KEY
- Uses specialized transformers and TRL versions pinned in setup.py for DeepSpeed x vLLM compatibility
- Supports multiple distributed training backends via accelerate configs

## Current Status (July 26, 2025)

### ✅ Completed Tasks:
1. **Performance Claims Updated**: Removed all specific PyTorch vs JAX performance claims from documentation
   - Updated: README.md, README_PYPI.md, BENCHMARKS.md, upload script model card
   - Now states: "Performance varies based on hardware, batch size, and workload"

2. **Model Conversion**: Successfully converted JAX model to sharded Orbax format
   - Original model: `/models/DiffuCoder-7B-JAX-original/params.pkl` (single pickle file)
   - Sharded model: `/models/DiffuCoder-7B-JAX/` (11.85 GB with Orbax checkpoint)
   - Used script: `convert_jax_to_sharded.py`

3. **Directory Structure**: Renamed to match HuggingFace convention
   - `dream-jax` → `DiffuCoder-7B-JAX-original`
   - `dream-jax-sharded` → `DiffuCoder-7B-JAX`

4. **Model Structure Verified**: 
   - JAX params are in FrozenDict format
   - Structure: `params['params']['DreamModel_0']` contains model weights
   - Total parameters: 7,615,487,488

### 🚧 In Progress:

1. **HuggingFace Upload**: Need manual upload due to token permissions
   - Model ready at: `/Users/amund/ml-diffucoder/models/DiffuCoder-7B-JAX`
   - Target repo: `atsentia/DiffuCoder-7B-JAX`
   - Upload script: `upload_to_huggingface.py -y`
   - Error: 403 Forbidden (token permissions issue)

2. **Import Issues**: Circular imports in jax_lm package
   - Main issue between `jax_lm.__init__.py` and model imports
   - Implemented lazy loading with `__getattr__` but needs testing

### 📋 TODO:

1. **Manual HuggingFace Upload**: Upload model files via web interface
2. **Fix Package Imports**: Resolve circular dependencies properly
3. **PyTorch/JAX Numerical Comparison**: Create proper parity test
4. **PyPI Publication**: After imports are fixed and tested
5. **Colab Testing**: 
   - Test on TPU v2/v3 and GPU (A100)
   - Fill in benchmark tables with real numbers
   - Update documentation with actual performance data

### 🔧 Key Scripts Created:

- `convert_jax_to_sharded.py` - Converts pickle to sharded Orbax format with rich progress
- `test_sharded_format.py` - Comprehensive format verification
- `verify_sharded_files.py` - Quick structure check
- `test_local_install.py` - Package installation test
- `test_pytorch_jax_parity.py` - Model comparison (needs work)
- `test_local_model_cpu.py` - CPU inference test
- `upload_to_huggingface.py` - HF upload with progress

### 📝 Important Notes:

- Force CPU mode: `os.environ["JAX_PLATFORM_NAME"] = "cpu"`
- Disable rich logging: `NO_RICH_LOGGING=1`
- HF token location: `/jax_lm/.env`
- Model files total: 11.85 GB (27 files including tokenizer)

### 🔐 Security Note:
- Never commit the .env file with HF token
- Token needs write permissions for model upload