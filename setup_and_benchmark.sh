#!/bin/bash
# Complete setup and benchmark script for JAX DiffuCoder

set -e

echo "=========================================="
echo "JAX DiffuCoder Setup and Benchmark"
echo "=========================================="

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
echo "--------------------------------------"
python3 -m venv jax_env
source jax_env/bin/activate

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
echo "---------------------------------"
pip install --upgrade pip

# Install JAX (CPU version for compatibility)
pip install jax jaxlib

# Install other dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface-hub
pip install flax optax orbax-checkpoint
pip install einops numpy psutil tqdm
pip install ml_collections chex

# Step 3: Download a small model for testing
echo ""
echo "Step 3: Preparing test model..."
echo "------------------------------"
# For demo, we'll use a smaller model or mock weights
mkdir -p models/test

# Step 4: Run quick demo benchmark
echo ""
echo "Step 4: Running quick demo benchmark..."
echo "--------------------------------------"
python jax_lm/benchmarks/quick_demo.py

# Step 5: Instructions for full benchmark
echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "To run the full benchmark with actual DiffuCoder weights:"
echo ""
echo "1. Download weights from HuggingFace:"
echo "   python jax_lm/scripts/download_weights.py \\"
echo "     --model-id apple/DiffuCoder-7B-Base \\"
echo "     --output-dir ./models/pytorch"
echo ""
echo "2. Convert to JAX format:"
echo "   python jax_lm/scripts/convert_pytorch_to_jax.py \\"
echo "     --pytorch-path ./models/pytorch/pytorch_weights \\"
echo "     --output-path ./models/jax"
echo ""
echo "3. Run full CPU benchmark:"
echo "   python jax_lm/benchmarks/hardware_benchmark.py \\"
echo "     --backend cpu \\"
echo "     --model-size small"
echo ""
echo "Note: Full model requires significant disk space and memory."