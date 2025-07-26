#!/bin/bash
# Script to download and benchmark the full DiffuCoder model

set -e

echo "============================================================"
echo "DiffuCoder Full Model Download and Benchmark Plan"
echo "============================================================"

# Check disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    echo "WARNING: Less than 50GB available. Full model requires ~30GB."
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Configuration
MODEL_ID="apple/DiffuCoder-7B-Base"
OUTPUT_DIR="./models/full"
JAX_OUTPUT_DIR="./models/full_jax"

echo ""
echo "Plan:"
echo "1. Download DiffuCoder-7B weights (~26GB)"
echo "2. Convert to JAX format"
echo "3. Run comprehensive benchmarks"
echo "4. Generate detailed report"

echo ""
echo "Estimated time: 30-60 minutes depending on internet speed"
echo ""
echo "Commands to run:"
echo ""
echo "# Step 1: Download weights"
echo "python jax_lm/scripts/download_weights.py \\"
echo "    --model-id $MODEL_ID \\"
echo "    --output-dir $OUTPUT_DIR"
echo ""
echo "# Step 2: Convert to JAX"
echo "python jax_lm/scripts/convert_pytorch_to_jax.py \\"
echo "    --pytorch-path $OUTPUT_DIR/pytorch_weights \\"
echo "    --output-path $JAX_OUTPUT_DIR"
echo ""
echo "# Step 3: Run PyTorch benchmark"
echo "python benchmarks/pytorch_cpu_benchmark.py \\"
echo "    --model-path $OUTPUT_DIR/pytorch_weights \\"
echo "    --output-file benchmark_results/pytorch_7b_results.json"
echo ""
echo "# Step 4: Run JAX benchmark"
echo "python jax_lm/benchmarks/hardware_benchmark.py \\"
echo "    --backend cpu \\"
echo "    --output-file benchmark_results/jax_7b_results.json"
echo ""
echo "# Step 5: Compare results"
echo "python jax_lm/benchmarks/compare_results.py \\"
echo "    --pytorch-results benchmark_results/pytorch_7b_results.json \\"
echo "    --jax-results benchmark_results/jax_7b_results.json \\"
echo "    --output-file benchmark_results/comparison_7b.txt"
echo ""
echo "Ready to proceed? This will download ~26GB of model weights."
echo "Type 'yes' to continue or 'no' to see memory-efficient alternatives:"
read -r response

if [ "$response" = "yes" ]; then
    echo "Starting download..."
    # Uncomment to actually run:
    # python jax_lm/scripts/download_weights.py --model-id $MODEL_ID --output-dir $OUTPUT_DIR
else
    echo ""
    echo "Alternative options:"
    echo "1. Use quantized models (int8/int4) to reduce memory by 4-8x"
    echo "2. Use model sharding to run on multiple machines"
    echo "3. Use cloud GPUs/TPUs for better performance"
    echo "4. Test with smaller variants first (e.g., 1B or 3B models)"
fi