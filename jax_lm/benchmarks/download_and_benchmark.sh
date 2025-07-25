#!/bin/bash
# Download models and run CPU benchmark comparing PyTorch and JAX

set -e

echo "=========================================="
echo "DiffuCoder CPU Benchmark Setup"
echo "=========================================="

# Configuration
MODEL_ID="apple/DiffuCoder-7B-Base"  # Using base model for benchmarking
PYTORCH_DIR="./models/pytorch"
JAX_DIR="./models/jax"
RESULTS_DIR="./benchmark_results"

# Create directories
mkdir -p $PYTORCH_DIR $JAX_DIR $RESULTS_DIR

# Step 1: Download PyTorch weights
echo ""
echo "Step 1: Downloading PyTorch weights..."
echo "--------------------------------------"
python jax_lm/scripts/download_weights.py \
    --model-id $MODEL_ID \
    --output-dir $PYTORCH_DIR

# Step 2: Convert to JAX format
echo ""
echo "Step 2: Converting PyTorch weights to JAX..."
echo "-------------------------------------------"
python jax_lm/scripts/convert_pytorch_to_jax.py \
    --pytorch-path $PYTORCH_DIR/pytorch_weights \
    --output-path $JAX_DIR

# Step 3: Run PyTorch CPU benchmark
echo ""
echo "Step 3: Running PyTorch CPU benchmark..."
echo "---------------------------------------"
python benchmarks/pytorch_cpu_benchmark.py \
    --model-path $PYTORCH_DIR/pytorch_weights \
    --output-file $RESULTS_DIR/pytorch_cpu_results.json

# Step 4: Run JAX CPU benchmark
echo ""
echo "Step 4: Running JAX CPU benchmark..."
echo "-----------------------------------"
python jax_lm/benchmarks/hardware_benchmark.py \
    --backend cpu \
    --model-size small \
    --output-file $RESULTS_DIR/jax_cpu_results.json

# Step 5: Run numerical parity test
echo ""
echo "Step 5: Running numerical parity test..."
echo "---------------------------------------"
python jax_lm/tests/test_numerical_parity.py \
    --pytorch-model $PYTORCH_DIR/pytorch_weights \
    --jax-model $JAX_DIR \
    --output-file $RESULTS_DIR/parity_results.json \
    --verbose

# Step 6: Generate comparison report
echo ""
echo "Step 6: Generating comparison report..."
echo "--------------------------------------"
python jax_lm/benchmarks/compare_results.py \
    --pytorch-results $RESULTS_DIR/pytorch_cpu_results.json \
    --jax-results $RESULTS_DIR/jax_cpu_results.json \
    --parity-results $RESULTS_DIR/parity_results.json \
    --output-file $RESULTS_DIR/comparison_report.txt

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Files created:"
echo "  - pytorch_cpu_results.json: PyTorch CPU benchmark results"
echo "  - jax_cpu_results.json: JAX CPU benchmark results"  
echo "  - parity_results.json: Numerical parity test results"
echo "  - comparison_report.txt: Summary comparison report"