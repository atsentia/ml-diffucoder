#!/usr/bin/env python3
"""Compare PyTorch and JAX benchmark results."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_results(file_path: str) -> Dict[str, Any]:
    """Load JSON results file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_benchmarks(pytorch_results: Dict, jax_results: Dict, parity_results: Dict) -> str:
    """Generate comparison report."""
    report = []
    report.append("=" * 80)
    report.append("PyTorch vs JAX DiffuCoder CPU Benchmark Comparison")
    report.append("=" * 80)
    
    # System info comparison
    report.append("\n## System Information")
    report.append("-" * 40)
    
    pt_sys = pytorch_results.get('system_info', {})
    jax_sys = jax_results.get('system_info', {})
    
    report.append(f"PyTorch:")
    report.append(f"  Version: {pt_sys.get('pytorch_version', 'N/A')}")
    report.append(f"  Device: {pt_sys.get('device', 'N/A')}")
    report.append(f"  CPU threads: {pt_sys.get('cpu_threads', 'N/A')}")
    report.append(f"  Model params: {pt_sys.get('param_count', 0) / 1e6:.2f}M")
    
    report.append(f"\nJAX:")
    report.append(f"  Backend: {jax_sys.get('backend', 'N/A')}")
    report.append(f"  Platform: {jax_sys.get('platform', 'N/A')}")
    report.append(f"  Device count: {jax_sys.get('device_count', 'N/A')}")
    report.append(f"  Model params: {jax_sys.get('param_count', 0) / 1e6:.2f}M")
    
    # Numerical parity results
    report.append("\n## Numerical Parity")
    report.append("-" * 40)
    
    if parity_results and 'summary' in parity_results:
        summary = parity_results['summary']
        report.append(f"Total tests: {summary.get('total_tests', 0)}")
        report.append(f"Passed: {summary.get('passed_tests', 0)}")
        report.append(f"Failed: {summary.get('failed_tests', 0)}")
        report.append(f"Pass rate: {summary.get('pass_rate', 0) * 100:.1f}%")
        
        # Show any failures
        if summary.get('failed_tests', 0) > 0:
            report.append("\nFailed tests:")
            layer_results = parity_results.get('layer_results', {})
            for layer, result in layer_results.items():
                if isinstance(result, dict) and not result.get('match', True):
                    report.append(f"  - {layer}: max diff = {result.get('max_abs_diff', 0):.2e}")
    
    # Forward pass comparison
    report.append("\n## Forward Pass Performance")
    report.append("-" * 40)
    
    pt_forward = pytorch_results.get('forward_pass', {})
    jax_forward = jax_results.get('forward_pass', {})
    
    report.append("\n| Batch | Seq Len | PyTorch (s) | JAX (s) | Speedup |")
    report.append("|-------|---------|-------------|---------|---------|")
    
    for batch_key in sorted(pt_forward.keys()):
        if batch_key in jax_forward:
            for seq_key in sorted(pt_forward[batch_key].keys()):
                if seq_key in jax_forward[batch_key]:
                    pt_time = pt_forward[batch_key][seq_key]['mean_time']
                    jax_time = jax_forward[batch_key][seq_key]['mean_time']
                    speedup = pt_time / jax_time
                    
                    batch_size = batch_key.split('_')[1]
                    seq_len = seq_key.split('_')[1]
                    
                    report.append(
                        f"| {batch_size:^5} | {seq_len:^7} | "
                        f"{pt_time:^11.3f} | {jax_time:^7.3f} | "
                        f"{speedup:^7.2f}x |"
                    )
    
    # Throughput comparison
    report.append("\n## Throughput (tokens/second)")
    report.append("-" * 40)
    
    report.append("\n| Batch | Seq Len | PyTorch | JAX | Improvement |")
    report.append("|-------|---------|---------|-----|-------------|")
    
    for batch_key in sorted(pt_forward.keys()):
        if batch_key in jax_forward:
            for seq_key in sorted(pt_forward[batch_key].keys()):
                if seq_key in jax_forward[batch_key]:
                    pt_tps = pt_forward[batch_key][seq_key]['tokens_per_second']
                    jax_tps = jax_forward[batch_key][seq_key]['tokens_per_second']
                    improvement = (jax_tps / pt_tps - 1) * 100
                    
                    batch_size = batch_key.split('_')[1]
                    seq_len = seq_key.split('_')[1]
                    
                    report.append(
                        f"| {batch_size:^5} | {seq_len:^7} | "
                        f"{pt_tps:^7.1f} | {jax_tps:^3.1f} | "
                        f"{improvement:^+10.1f}% |"
                    )
    
    # Memory comparison
    report.append("\n## Memory Usage")
    report.append("-" * 40)
    
    pt_mem = pytorch_results.get('memory', {})
    jax_mem = jax_results.get('memory', {})
    
    if pt_mem and jax_mem:
        report.append(f"\nModel parameters:")
        report.append(f"  PyTorch: {pt_mem.get('model_params_mb', 0):.1f} MB")
        report.append(f"  JAX: {jax_mem.get('model_params_mb', 0):.1f} MB")
        
        report.append(f"\nPeak memory increase by batch size:")
        for batch in [1, 2, 4, 8]:
            pt_increase = pt_mem.get(f'batch_{batch}_increase_mb', 0)
            jax_increase = jax_mem.get(f'batch_{batch}_increase_mb', 0)
            if pt_increase > 0 or jax_increase > 0:
                report.append(f"  Batch {batch}: PyTorch +{pt_increase:.1f} MB, JAX +{jax_increase:.1f} MB")
    
    # Summary
    report.append("\n## Summary")
    report.append("-" * 40)
    
    # Calculate average speedup
    speedups = []
    for batch_key in pt_forward.keys():
        if batch_key in jax_forward:
            for seq_key in pt_forward[batch_key].keys():
                if seq_key in jax_forward[batch_key]:
                    pt_time = pt_forward[batch_key][seq_key]['mean_time']
                    jax_time = jax_forward[batch_key][seq_key]['mean_time']
                    speedups.append(pt_time / jax_time)
    
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        report.append(f"\nAverage speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 1.0:
            report.append("JAX is faster than PyTorch on average! 🚀")
        else:
            report.append("PyTorch is faster than JAX on average.")
    
    # Numerical accuracy
    if parity_results and 'summary' in parity_results:
        pass_rate = parity_results['summary'].get('pass_rate', 0)
        if pass_rate == 1.0:
            report.append("\nNumerical parity: PERFECT ✅")
        elif pass_rate >= 0.95:
            report.append(f"\nNumerical parity: GOOD ({pass_rate*100:.1f}%) ✓")
        else:
            report.append(f"\nNumerical parity: NEEDS ATTENTION ({pass_rate*100:.1f}%) ⚠️")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and JAX benchmark results"
    )
    parser.add_argument(
        "--pytorch-results",
        type=str,
        required=True,
        help="Path to PyTorch benchmark results JSON",
    )
    parser.add_argument(
        "--jax-results",
        type=str,
        required=True,
        help="Path to JAX benchmark results JSON",
    )
    parser.add_argument(
        "--parity-results",
        type=str,
        help="Path to numerical parity test results JSON",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="comparison_report.txt",
        help="Output file for comparison report",
    )
    
    args = parser.parse_args()
    
    # Load results
    pytorch_results = load_results(args.pytorch_results)
    jax_results = load_results(args.jax_results)
    parity_results = load_results(args.parity_results) if args.parity_results else {}
    
    # Generate comparison report
    report = compare_benchmarks(pytorch_results, jax_results, parity_results)
    
    # Save report
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(report)
    
    # Also print to console
    print(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()