#!/usr/bin/env python3
"""Layer-wise numerical parity testing between PyTorch and JAX DiffuCoder.

This test loads the same weights in both frameworks and compares outputs
layer by layer to ensure numerical correctness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional


class LayerParityTester:
    """Test numerical parity between PyTorch and JAX implementations."""
    
    def __init__(
        self,
        pytorch_model_path: str = "./models/diffucoder-7b-complete",
        jax_model_path: str = "./models/dream-jax",
        tolerance: float = 1e-5,
        verbose: bool = True,
    ):
        self.pytorch_model_path = Path(pytorch_model_path)
        self.jax_model_path = Path(jax_model_path)
        self.tolerance = tolerance
        self.verbose = verbose
        self.results = []
        
    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(message)
    
    def compare_tensors(
        self,
        name: str,
        pytorch_tensor: np.ndarray,
        jax_tensor: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compare PyTorch and JAX tensors."""
        if rtol is None:
            rtol = self.tolerance
        if atol is None:
            atol = self.tolerance
        
        # Ensure numpy arrays
        if hasattr(pytorch_tensor, "numpy"):
            pytorch_tensor = pytorch_tensor.detach().cpu().numpy()
        if hasattr(jax_tensor, "numpy"):
            jax_tensor = np.array(jax_tensor)
        
        # Check shapes
        if pytorch_tensor.shape != jax_tensor.shape:
            return {
                "name": name,
                "match": False,
                "error": "shape_mismatch",
                "pytorch_shape": pytorch_tensor.shape,
                "jax_shape": jax_tensor.shape,
            }
        
        # Compute differences
        abs_diff = np.abs(pytorch_tensor - jax_tensor)
        rel_diff = abs_diff / (np.abs(pytorch_tensor) + 1e-8)
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        # Check if close
        is_close = np.allclose(pytorch_tensor, jax_tensor, rtol=rtol, atol=atol)
        
        result = {
            "name": name,
            "match": is_close,
            "shape": pytorch_tensor.shape,
            "max_abs_diff": float(max_abs_diff),
            "max_rel_diff": float(max_rel_diff),
            "mean_abs_diff": float(mean_abs_diff),
        }
        
        if not is_close and self.verbose:
            # Find location of maximum difference
            max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            result["max_diff_location"] = max_idx
            result["pytorch_value_at_max"] = float(pytorch_tensor[max_idx])
            result["jax_value_at_max"] = float(jax_tensor[max_idx])
        
        return result
    
    def test_embedding_layer(self) -> Dict[str, Any]:
        """Test embedding layer parity."""
        self.log("\n🔍 Testing Embedding Layer...")
        
        try:
            # Create test input
            test_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
            
            # PyTorch embedding
            import torch
            # Load PyTorch embedding weights
            pytorch_embed_path = self.pytorch_model_path / "model-00001-of-00004.safetensors"
            if pytorch_embed_path.exists():
                from safetensors import safe_open
                with safe_open(pytorch_embed_path, framework="pt") as f:
                    if "model.embed_tokens.weight" in f.keys():
                        pytorch_embed_weight = f.get_tensor("model.embed_tokens.weight")
                    else:
                        self.log("  ⚠️  PyTorch embedding weights not found")
                        return {"error": "pytorch_weights_not_found"}
            
            # JAX embedding
            from jax_lm.models.dream import DreamConfig, DreamForCausalLM
            
            # Load JAX config
            import json
            with open(self.jax_model_path / "config.json") as f:
                config_dict = json.load(f)
            
            config = DreamConfig(**{k: v for k, v in config_dict.items() if hasattr(DreamConfig, k)})
            model = DreamForCausalLM(config=config)
            
            # Load JAX params
            import pickle
            with open(self.jax_model_path / "params.pkl", "rb") as f:
                jax_params = pickle.load(f)
            
            # Extract embedding weights
            jax_embed_weight = jax_params["params"]["embed_tokens"]["embedding"]
            
            # Compare weights first
            weight_result = self.compare_tensors(
                "embedding_weights",
                pytorch_embed_weight.numpy(),
                jax_embed_weight,
            )
            
            self.log(f"  Embedding weights match: {weight_result['match']}")
            if not weight_result['match']:
                self.log(f"    Max diff: {weight_result['max_abs_diff']:.2e}")
            
            # Test forward pass
            torch_input = torch.tensor(test_ids, dtype=torch.long)
            torch_output = torch.nn.functional.embedding(torch_input, pytorch_embed_weight)
            
            jax_output = jax_embed_weight[test_ids]
            
            output_result = self.compare_tensors(
                "embedding_output",
                torch_output.numpy(),
                jax_output,
            )
            
            self.log(f"  Embedding output match: {output_result['match']}")
            if not output_result['match']:
                self.log(f"    Max diff: {output_result['max_abs_diff']:.2e}")
            
            return {
                "weight_comparison": weight_result,
                "output_comparison": output_result,
                "overall_match": weight_result['match'] and output_result['match'],
            }
            
        except Exception as e:
            self.log(f"  ❌ Embedding test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def test_rmsnorm_layer(self, layer_idx: int = 0) -> Dict[str, Any]:
        """Test RMSNorm layer parity."""
        self.log(f"\n🔍 Testing RMSNorm Layer {layer_idx}...")
        
        try:
            # Create test input
            batch_size, seq_len, hidden_size = 1, 8, 3584
            test_input = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
            eps = 1e-6
            
            # Load weights
            import torch
            from safetensors import safe_open
            
            # Find PyTorch RMSNorm weight
            weight_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            pytorch_weight = None
            
            for i in range(1, 5):
                safetensor_path = self.pytorch_model_path / f"model-0000{i}-of-00004.safetensors"
                with safe_open(safetensor_path, framework="pt") as f:
                    if weight_key in f.keys():
                        pytorch_weight = f.get_tensor(weight_key)
                        break
            
            if pytorch_weight is None:
                self.log(f"  ⚠️  PyTorch RMSNorm weight not found for layer {layer_idx}")
                return {"error": "pytorch_weight_not_found"}
            
            # JAX weight
            import pickle
            with open(self.jax_model_path / "params.pkl", "rb") as f:
                jax_params = pickle.load(f)
            
            jax_weight = jax_params["params"]["layers"][str(layer_idx)]["input_layernorm"]["scale"]
            
            # Compare weights
            weight_result = self.compare_tensors(
                f"rmsnorm_weight_layer_{layer_idx}",
                pytorch_weight.numpy(),
                jax_weight,
            )
            
            self.log(f"  RMSNorm weights match: {weight_result['match']}")
            
            # Test forward pass
            # PyTorch RMSNorm
            torch_input = torch.tensor(test_input)
            torch_variance = torch_input.pow(2).mean(-1, keepdim=True)
            torch_input_norm = torch_input * torch.rsqrt(torch_variance + eps)
            torch_output = pytorch_weight * torch_input_norm
            
            # JAX RMSNorm
            jax_input = jnp.array(test_input)
            jax_variance = jnp.mean(jax_input ** 2, axis=-1, keepdims=True)
            jax_input_norm = jax_input * jax.lax.rsqrt(jax_variance + eps)
            jax_output = jax_weight * jax_input_norm
            
            output_result = self.compare_tensors(
                f"rmsnorm_output_layer_{layer_idx}",
                torch_output.numpy(),
                jax_output,
                rtol=1e-4,  # Slightly relaxed for RMSNorm
            )
            
            self.log(f"  RMSNorm output match: {output_result['match']}")
            if not output_result['match']:
                self.log(f"    Max diff: {output_result['max_abs_diff']:.2e}")
            
            return {
                "weight_comparison": weight_result,
                "output_comparison": output_result,
                "overall_match": weight_result['match'] and output_result['match'],
            }
            
        except Exception as e:
            self.log(f"  ❌ RMSNorm test failed: {e}")
            return {"error": str(e)}
    
    def test_attention_projection(self, layer_idx: int = 0, proj_type: str = "q") -> Dict[str, Any]:
        """Test attention projection (Q, K, V, O) parity."""
        self.log(f"\n🔍 Testing Attention {proj_type.upper()}-projection Layer {layer_idx}...")
        
        try:
            # Create test input
            batch_size, seq_len = 1, 8
            hidden_size = 3584  # From config
            test_input = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
            
            # Load weights
            import torch
            from safetensors import safe_open
            
            # Map projection types
            proj_map = {
                "q": "q_proj",
                "k": "k_proj", 
                "v": "v_proj",
                "o": "o_proj",
            }
            
            weight_key = f"model.layers.{layer_idx}.self_attn.{proj_map[proj_type]}.weight"
            pytorch_weight = None
            
            for i in range(1, 5):
                safetensor_path = self.pytorch_model_path / f"model-0000{i}-of-00004.safetensors"
                with safe_open(safetensor_path, framework="pt") as f:
                    if weight_key in f.keys():
                        pytorch_weight = f.get_tensor(weight_key)
                        break
            
            if pytorch_weight is None:
                self.log(f"  ⚠️  PyTorch weight not found")
                return {"error": "pytorch_weight_not_found"}
            
            # JAX weight
            import pickle
            with open(self.jax_model_path / "params.pkl", "rb") as f:
                jax_params = pickle.load(f)
            
            jax_weight = jax_params["params"]["layers"][str(layer_idx)]["self_attn"][proj_map[proj_type]]["kernel"]
            
            # Note: JAX uses different weight layout (transposed)
            # PyTorch: [out_features, in_features]
            # JAX: [in_features, out_features]
            
            # Compare weights (after transposing)
            weight_result = self.compare_tensors(
                f"attention_{proj_type}_weight_layer_{layer_idx}",
                pytorch_weight.T.numpy(),  # Transpose PyTorch weight
                jax_weight,
            )
            
            self.log(f"  Weight shapes - PyTorch: {pytorch_weight.shape}, JAX: {jax_weight.shape}")
            self.log(f"  Weights match: {weight_result['match']}")
            
            # Test forward pass
            torch_input = torch.tensor(test_input)
            torch_output = torch.nn.functional.linear(torch_input, pytorch_weight)
            
            jax_input = jnp.array(test_input)
            jax_output = jnp.dot(jax_input, jax_weight)
            
            output_result = self.compare_tensors(
                f"attention_{proj_type}_output_layer_{layer_idx}",
                torch_output.numpy(),
                jax_output,
            )
            
            self.log(f"  Output match: {output_result['match']}")
            if not output_result['match']:
                self.log(f"    Max diff: {output_result['max_abs_diff']:.2e}")
            
            return {
                "weight_comparison": weight_result,
                "output_comparison": output_result,
                "overall_match": weight_result['match'] and output_result['match'],
            }
            
        except Exception as e:
            self.log(f"  ❌ Attention projection test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all parity tests."""
        print("=" * 70)
        print("PyTorch vs JAX Layer-wise Numerical Parity Test")
        print("=" * 70)
        print(f"PyTorch model: {self.pytorch_model_path}")
        print(f"JAX model: {self.jax_model_path}")
        print(f"Tolerance: {self.tolerance}")
        print("=" * 70)
        
        results = {}
        
        # Test embedding
        results["embedding"] = self.test_embedding_layer()
        
        # Test first few layers
        for layer_idx in range(2):  # Test first 2 layers
            # RMSNorm
            results[f"rmsnorm_layer_{layer_idx}"] = self.test_rmsnorm_layer(layer_idx)
            
            # Attention projections
            for proj in ["q", "k", "v", "o"]:
                results[f"attention_{proj}_layer_{layer_idx}"] = self.test_attention_projection(
                    layer_idx, proj
                )
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(
            1 for r in results.values() 
            if isinstance(r, dict) and r.get("overall_match", False)
        )
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass rate: {passed_tests/total_tests*100:.1f}%")
        
        # Print failures
        failures = [
            (name, result) for name, result in results.items()
            if isinstance(result, dict) and not result.get("overall_match", False)
        ]
        
        if failures:
            print("\nFailed tests:")
            for name, result in failures:
                if "error" in result:
                    print(f"  - {name}: {result['error']}")
                else:
                    print(f"  - {name}")
        
        return {
            "results": results,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "pass_rate": passed_tests/total_tests if total_tests > 0 else 0,
            }
        }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test layer-wise numerical parity between PyTorch and JAX"
    )
    parser.add_argument(
        "--pytorch-model",
        default="./models/diffucoder-7b-complete",
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--jax-model",
        default="./models/dream-jax",
        help="Path to JAX model",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Numerical tolerance",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available. Install with: pip install torch")
        return 1
    
    # Run tests
    tester = LayerParityTester(
        pytorch_model_path=args.pytorch_model,
        jax_model_path=args.jax_model,
        tolerance=args.tolerance,
        verbose=not args.quiet,
    )
    
    results = tester.run_all_tests()
    
    # Save results
    import json
    with open("layer_parity_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to layer_parity_results.json")
    
    return 0 if results["summary"]["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())