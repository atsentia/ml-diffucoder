# CLAUDE.md - JAX/Flax Implementation Status

This file provides guidance for working with the JAX/Flax implementation of DiffuCoder.

## Current Status (July 26, 2025)

### ✅ Completed Tasks

1. **Performance Claims Updated**
   - Removed all specific PyTorch vs JAX performance claims from documentation
   - Updated: README.md, README_PYPI.md, BENCHMARKS.md, upload script model card
   - Now states: "Performance varies based on hardware, batch size, and workload"

2. **Model Conversion to Sharded Format**
   - Successfully converted JAX model from single pickle to Orbax sharded format
   - Original: `models/DiffuCoder-7B-JAX-original/params.pkl` (15.23 GB)
   - Sharded: `models/DiffuCoder-7B-JAX/` (11.85 GB, 27 files)
   - ~22% size reduction due to Orbax's efficient compression

3. **Directory Structure**
   - Renamed to match HuggingFace convention
   - `dream-jax` → `DiffuCoder-7B-JAX-original`
   - `dream-jax-sharded` → `DiffuCoder-7B-JAX`

4. **Testing Infrastructure**
   - Created comprehensive test scripts for format verification
   - CPU inference forced to avoid JAX Metal issues on macOS
   - Rich logging with disable option (`NO_RICH_LOGGING=1`)

### 🚧 Known Issues

1. **Circular Imports**
   - Issue between `jax_lm.__init__.py` and model imports
   - Implemented lazy loading with `__getattr__` but needs proper testing
   - This blocks PyPI publication

2. **HuggingFace Upload**
   - Model ready at: `models/DiffuCoder-7B-JAX/`
   - Target repo: `atsentia/DiffuCoder-7B-JAX`
   - Current blocker: 403 Forbidden (needs manual upload with proper permissions)

### 📋 Pending Tasks

1. **Manual HuggingFace Upload**
   - Upload the sharded model files via web interface
   - Total size: 11.85 GB (27 files including tokenizer)

2. **Fix Package Imports**
   - Resolve circular dependencies properly
   - Test package installation

3. **PyTorch/JAX Numerical Comparison**
   - Create proper parity test comparing outputs

4. **PyPI Publication**
   - After imports are fixed and tested

5. **Performance Benchmarking**
   - Test on Google Colab with TPU v2/v3 and GPU (A100)
   - Fill benchmark tables with real numbers
   - Update documentation with actual performance data

## Key Configuration

### Force CPU Mode (macOS)
```python
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
```

### Disable Rich Logging
```bash
export NO_RICH_LOGGING=1
```

### Model Structure
- JAX params are in FrozenDict format
- Structure: `params['params']['DreamModel_0']` contains model weights
- Total parameters: 7,615,487,488

## Important Scripts

- `convert_jax_to_sharded.py` - Converts pickle to Orbax sharded format
- `test_sharded_format.py` - Comprehensive format verification
- `verify_sharded_files.py` - Quick structure check
- `test_local_install.py` - Package installation test
- `test_pytorch_jax_parity.py` - Model comparison (needs work)
- `test_local_model_cpu.py` - CPU inference test
- `upload_to_huggingface.py` - HF upload with progress

## Notes

- The .env file in jax_lm/.env contains the HuggingFace token
- Model weights are excluded from git via .gitignore
- Pre-commit hooks disabled for faster commits