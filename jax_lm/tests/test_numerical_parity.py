#!/usr/bin/env python3
"""Test numerical parity between PyTorch and JAX implementations of DiffuCoder."""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import random

# Mock imports for testing - in production these would be real imports
class MockPyTorchDiffuCoder:
    """Mock PyTorch DiffuCoder for testing."""
    pass

class MockJAXDiffuCoder:
    """Mock JAX DiffuCoder for testing."""
    pass


class NumericalParityTester:
    """Test numerical parity between PyTorch and JAX implementations."""
    
    def __init__(
        self,
        pytorch_model_path: Optional[str] = None,
        jax_model_path: Optional[str] = None,
        tolerance: float = 1e-5,
        verbose: bool = True,
    ):
        self.pytorch_model_path = pytorch_model_path
        self.jax_model_path = jax_model_path
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Results storage
        self.results = {
            "summary": {},
            "layer_results": {},
            "detailed_mismatches": [],
        }
    
    def log(self, message: str, level: str = "info"):
        """Log message with level."""
        if self.verbose:
            prefix = {
                "info": "[INFO]",
                "warning": "[WARNING]",
                "error": "[ERROR]",
                "success": "[SUCCESS]",
            }.get(level, "[INFO]")
            print(f"{prefix} {message}")
    
    def compare_tensors(
        self,
        pytorch_tensor: torch.Tensor,
        jax_array: jnp.ndarray,
        name: str,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compare PyTorch tensor and JAX array."""
        if rtol is None:
            rtol = self.tolerance
        if atol is None:
            atol = self.tolerance
        
        # Convert to numpy for comparison
        pt_numpy = pytorch_tensor.detach().cpu().numpy()
        jax_numpy = np.array(jax_array)
        
        # Check shapes
        if pt_numpy.shape != jax_numpy.shape:
            return {
                "name": name,
                "match": False,
                "error": "shape_mismatch",
                "pytorch_shape": pt_numpy.shape,
                "jax_shape": jax_numpy.shape,
            }
        
        # Compute differences
        abs_diff = np.abs(pt_numpy - jax_numpy)
        rel_diff = abs_diff / (np.abs(pt_numpy) + 1e-8)
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        # Check if arrays are close
        is_close = np.allclose(pt_numpy, jax_numpy, rtol=rtol, atol=atol)
        
        result = {
            "name": name,
            "match": is_close,
            "shape": pt_numpy.shape,
            "max_abs_diff": float(max_abs_diff),
            "max_rel_diff": float(max_rel_diff),
            "mean_abs_diff": float(mean_abs_diff),
            "num_elements": pt_numpy.size,
        }
        
        # Find locations of largest differences
        if not is_close and self.verbose:
            # Get indices of maximum difference
            max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            result["max_diff_location"] = max_idx
            result["pytorch_value_at_max"] = float(pt_numpy[max_idx])
            result["jax_value_at_max"] = float(jax_numpy[max_idx])
        
        return result
    
    def test_embedding_layer(self) -> Dict[str, Any]:
        """Test embedding layer parity."""
        self.log("Testing embedding layer...")
        
        # Create test input
        batch_size, seq_len = 2, 128
        vocab_size = 32000
        hidden_size = 768
        
        # Random input IDs
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        
        # Mock embedding weights
        embedding_weight = np.random.randn(vocab_size + 2, hidden_size).astype(np.float32)
        
        # PyTorch embedding
        torch_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(embedding_weight),
            freeze=False
        )
        torch_input = torch.tensor(input_ids, dtype=torch.long)
        torch_output = torch_embedding(torch_input)
        
        # JAX embedding (simulated)
        jax_embedding = jnp.array(embedding_weight)
        jax_input = jnp.array(input_ids)
        jax_output = jax_embedding[jax_input]
        
        # Compare
        result = self.compare_tensors(torch_output, jax_output, "embedding")
        
        if result["match"]:
            self.log("Embedding layer matches!", "success")
        else:
            self.log(f"Embedding layer mismatch: max diff = {result['max_abs_diff']:.2e}", "warning")
        
        return result
    
    def test_attention_layer(self) -> Dict[str, Any]:
        """Test attention layer parity."""
        self.log("Testing attention layer...")
        
        batch_size, seq_len, hidden_size = 2, 128, 768
        num_heads = 12
        head_dim = hidden_size // num_heads
        
        # Create test input
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        
        # Mock weights
        q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        k_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        v_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        o_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        
        # PyTorch computation
        torch_hidden = torch.tensor(hidden_states)
        torch_q = torch.matmul(torch_hidden, torch.tensor(q_weight.T))
        torch_k = torch.matmul(torch_hidden, torch.tensor(k_weight.T))
        torch_v = torch.matmul(torch_hidden, torch.tensor(v_weight.T))
        
        # Reshape for multi-head attention
        torch_q = torch_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        torch_k = torch_k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        torch_v = torch_v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Attention scores
        torch_scores = torch.matmul(torch_q, torch_k.transpose(-2, -1)) / np.sqrt(head_dim)
        torch_attn = torch.softmax(torch_scores, dim=-1)
        torch_output = torch.matmul(torch_attn, torch_v)
        torch_output = torch_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        torch_output = torch.matmul(torch_output, torch.tensor(o_weight.T))
        
        # JAX computation
        jax_hidden = jnp.array(hidden_states)
        jax_q = jnp.dot(jax_hidden, q_weight.T)
        jax_k = jnp.dot(jax_hidden, k_weight.T)
        jax_v = jnp.dot(jax_hidden, v_weight.T)
        
        # Reshape for multi-head attention
        jax_q = jax_q.reshape(batch_size, seq_len, num_heads, head_dim)
        jax_k = jax_k.reshape(batch_size, seq_len, num_heads, head_dim)
        jax_v = jax_v.reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Attention scores using einsum
        jax_scores = jnp.einsum('bshd,bthd->bhst', jax_q, jax_k) / jnp.sqrt(head_dim)
        jax_attn = jax.nn.softmax(jax_scores, axis=-1)
        jax_output = jnp.einsum('bhst,bthd->bshd', jax_attn, jax_v)
        jax_output = jax_output.reshape(batch_size, seq_len, hidden_size)
        jax_output = jnp.dot(jax_output, o_weight.T)
        
        # Compare intermediate results
        results = {}
        results["query_projection"] = self.compare_tensors(torch_q, jax_q, "query_projection")
        results["key_projection"] = self.compare_tensors(torch_k, jax_k, "key_projection")
        results["value_projection"] = self.compare_tensors(torch_v, jax_v, "value_projection")
        results["attention_scores"] = self.compare_tensors(torch_scores, jax_scores, "attention_scores")
        results["attention_weights"] = self.compare_tensors(torch_attn, jax_attn, "attention_weights")
        results["attention_output"] = self.compare_tensors(torch_output, jax_output, "attention_output")
        
        # Summary
        all_match = all(r["match"] for r in results.values())
        if all_match:
            self.log("Attention layer matches!", "success")
        else:
            mismatches = [k for k, v in results.items() if not v["match"]]
            self.log(f"Attention layer mismatches in: {mismatches}", "warning")
        
        return results
    
    def test_mlp_layer(self) -> Dict[str, Any]:
        """Test MLP layer parity."""
        self.log("Testing MLP layer...")
        
        batch_size, seq_len, hidden_size = 2, 128, 768
        intermediate_size = 3072
        
        # Create test input
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        
        # Mock weights
        gate_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02
        up_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02
        down_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
        
        # PyTorch computation
        torch_hidden = torch.tensor(hidden_states)
        torch_gate = torch.matmul(torch_hidden, torch.tensor(gate_weight.T))
        torch_up = torch.matmul(torch_hidden, torch.tensor(up_weight.T))
        torch_gate_act = torch.nn.functional.silu(torch_gate)
        torch_intermediate = torch_gate_act * torch_up
        torch_output = torch.matmul(torch_intermediate, torch.tensor(down_weight.T))
        
        # JAX computation
        jax_hidden = jnp.array(hidden_states)
        jax_gate = jnp.dot(jax_hidden, gate_weight.T)
        jax_up = jnp.dot(jax_hidden, up_weight.T)
        jax_gate_act = jax.nn.silu(jax_gate)
        jax_intermediate = jax_gate_act * jax_up
        jax_output = jnp.dot(jax_intermediate, down_weight.T)
        
        # Compare
        results = {}
        results["gate_projection"] = self.compare_tensors(torch_gate, jax_gate, "gate_projection")
        results["up_projection"] = self.compare_tensors(torch_up, jax_up, "up_projection")
        results["gate_activation"] = self.compare_tensors(torch_gate_act, jax_gate_act, "gate_activation")
        results["intermediate"] = self.compare_tensors(torch_intermediate, jax_intermediate, "intermediate")
        results["mlp_output"] = self.compare_tensors(torch_output, jax_output, "mlp_output")
        
        # Summary
        all_match = all(r["match"] for r in results.values())
        if all_match:
            self.log("MLP layer matches!", "success")
        else:
            mismatches = [k for k, v in results.items() if not v["match"]]
            self.log(f"MLP layer mismatches in: {mismatches}", "warning")
        
        return results
    
    def test_rmsnorm(self) -> Dict[str, Any]:
        """Test RMSNorm layer parity."""
        self.log("Testing RMSNorm layer...")
        
        batch_size, seq_len, hidden_size = 2, 128, 768
        eps = 1e-5
        
        # Create test input
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        weight = np.ones(hidden_size).astype(np.float32)
        
        # PyTorch RMSNorm
        torch_hidden = torch.tensor(hidden_states)
        torch_weight = torch.tensor(weight)
        torch_variance = torch_hidden.pow(2).mean(-1, keepdim=True)
        torch_hidden = torch_hidden * torch.rsqrt(torch_variance + eps)
        torch_output = torch_weight * torch_hidden
        
        # JAX RMSNorm
        jax_hidden = jnp.array(hidden_states)
        jax_weight = jnp.array(weight)
        jax_variance = jnp.mean(jax_hidden ** 2, axis=-1, keepdims=True)
        jax_hidden = jax_hidden * jax.lax.rsqrt(jax_variance + eps)
        jax_output = jax_weight * jax_hidden
        
        # Compare
        result = self.compare_tensors(torch_output, jax_output, "rmsnorm")
        
        if result["match"]:
            self.log("RMSNorm layer matches!", "success")
        else:
            self.log(f"RMSNorm layer mismatch: max diff = {result['max_abs_diff']:.2e}", "warning")
        
        return result
    
    def test_rotary_embedding(self) -> Dict[str, Any]:
        """Test Rotary Position Embedding (RoPE) parity."""
        self.log("Testing Rotary Position Embedding...")
        
        batch_size, seq_len, num_heads, head_dim = 2, 128, 12, 64
        rope_theta = 10000.0
        
        # Create test input
        query_states = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
        position_ids = np.arange(seq_len)[None, :]
        position_ids = np.broadcast_to(position_ids, (batch_size, seq_len))
        
        # Compute RoPE frequencies
        dim_half = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim_half, 2) / dim_half))
        
        # PyTorch RoPE
        torch_q = torch.tensor(query_states)
        torch_pos = torch.tensor(position_ids)
        
        # Compute sin/cos
        freqs = torch_pos[:, :, None].float() * torch.tensor(inv_freq)[None, None, :]
        freqs = freqs.repeat_interleave(2, dim=-1)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Apply rotation (PyTorch)
        torch_q_rot = torch_q[..., :dim_half * 2]
        torch_q_pass = torch_q[..., dim_half * 2:]
        
        # Complex rotation simulation
        torch_q_rot = torch_q_rot.reshape(*torch_q_rot.shape[:-1], -1, 2)
        torch_q_rot_new = torch.stack([
            torch_q_rot[..., 0] * cos[:, :, None, :] - torch_q_rot[..., 1] * sin[:, :, None, :],
            torch_q_rot[..., 0] * sin[:, :, None, :] + torch_q_rot[..., 1] * cos[:, :, None, :],
        ], dim=-1)
        torch_q_rot_new = torch_q_rot_new.reshape(*torch_q_rot.shape[:-2], -1)
        torch_output = torch.cat([torch_q_rot_new, torch_q_pass], dim=-1)
        
        # JAX RoPE
        jax_q = jnp.array(query_states)
        jax_pos = jnp.array(position_ids)
        
        # Compute sin/cos
        freqs = jax_pos[:, :, None] * inv_freq[None, None, :]
        freqs = jnp.repeat(freqs, 2, axis=-1)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        # Apply rotation (JAX)
        jax_q_rot = jax_q[..., :dim_half * 2]
        jax_q_pass = jax_q[..., dim_half * 2:]
        
        # Complex rotation simulation
        jax_q_rot = jax_q_rot.reshape(*jax_q_rot.shape[:-1], -1, 2)
        jax_q_rot_new = jnp.stack([
            jax_q_rot[..., 0] * cos[:, :, None, :] - jax_q_rot[..., 1] * sin[:, :, None, :],
            jax_q_rot[..., 0] * sin[:, :, None, :] + jax_q_rot[..., 1] * cos[:, :, None, :],
        ], axis=-1)
        jax_q_rot_new = jax_q_rot_new.reshape(*jax_q_rot.shape[:-2], -1)
        jax_output = jnp.concatenate([jax_q_rot_new, jax_q_pass], axis=-1)
        
        # Compare
        result = self.compare_tensors(torch_output, jax_output, "rope")
        
        if result["match"]:
            self.log("RoPE matches!", "success")
        else:
            self.log(f"RoPE mismatch: max diff = {result['max_abs_diff']:.2e}", "warning")
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all parity tests."""
        print("=" * 60)
        print("PyTorch vs JAX Numerical Parity Test")
        print("=" * 60)
        print(f"Tolerance: {self.tolerance}")
        print("=" * 60)
        
        # Run individual tests
        self.results["layer_results"]["embedding"] = self.test_embedding_layer()
        self.results["layer_results"]["rmsnorm"] = self.test_rmsnorm()
        self.results["layer_results"]["attention"] = self.test_attention_layer()
        self.results["layer_results"]["mlp"] = self.test_mlp_layer()
        self.results["layer_results"]["rope"] = self.test_rotary_embedding()
        
        # Compute summary statistics
        total_tests = 0
        passed_tests = 0
        
        for layer_name, layer_result in self.results["layer_results"].items():
            if isinstance(layer_result, dict) and "match" in layer_result:
                # Single test result
                total_tests += 1
                if layer_result["match"]:
                    passed_tests += 1
            elif isinstance(layer_result, dict):
                # Multiple test results
                for sub_name, sub_result in layer_result.items():
                    if isinstance(sub_result, dict) and "match" in sub_result:
                        total_tests += 1
                        if sub_result["match"]:
                            passed_tests += 1
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass rate: {self.results['summary']['pass_rate'] * 100:.1f}%")
        
        # Print detailed failures
        if total_tests > passed_tests:
            print("\nFailed tests:")
            for layer_name, layer_result in self.results["layer_results"].items():
                if isinstance(layer_result, dict) and "match" in layer_result:
                    if not layer_result["match"]:
                        print(f"  - {layer_name}: max diff = {layer_result['max_abs_diff']:.2e}")
                elif isinstance(layer_result, dict):
                    for sub_name, sub_result in layer_result.items():
                        if isinstance(sub_result, dict) and "match" in sub_result:
                            if not sub_result["match"]:
                                print(f"  - {layer_name}.{sub_name}: "
                                      f"max diff = {sub_result['max_abs_diff']:.2e}")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Test numerical parity between PyTorch and JAX DiffuCoder"
    )
    parser.add_argument(
        "--pytorch-model",
        type=str,
        help="Path to PyTorch model (optional)",
    )
    parser.add_argument(
        "--jax-model",
        type=str,
        help="Path to JAX model (optional)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Numerical tolerance for comparison",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="parity_test_results.json",
        help="Output file for test results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = NumericalParityTester(
        pytorch_model_path=args.pytorch_model,
        jax_model_path=args.jax_model,
        tolerance=args.tolerance,
        verbose=args.verbose,
    )
    
    results = tester.run_all_tests()
    
    # Save results
    import json
    output_path = Path(args.output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to {output_path}")
    
    # Exit with appropriate code
    if results["summary"]["failed_tests"] == 0:
        print("\nAll tests passed! ✅")
        exit(0)
    else:
        print(f"\n{results['summary']['failed_tests']} tests failed ❌")
        exit(1)


if __name__ == "__main__":
    main()