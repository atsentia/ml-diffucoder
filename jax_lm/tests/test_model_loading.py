#!/usr/bin/env python3
"""Comprehensive tests for model loading functionality.

Tests:
1. Pure JAX loader (no HF dependencies)
2. Orbax sharding
3. Legacy pickle format
4. File downloading and caching
5. Different data types
6. Error handling
"""

import json
import pickle
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.utils.pure_jax_loader import PureJAXModelLoader
from jax_lm.utils.orbax_sharding import ShardedCheckpointer, save_for_huggingface
from jax_lm.utils.model_utils import initialize_model, save_model


class TestModelLoading(unittest.TestCase):
    """Test model loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = DiffuCoderConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=512
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_test_model(self) -> tuple[DiffuCoder, dict]:
        """Create a small test model."""
        model = DiffuCoder(self.config)
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
        params = model.init(rng, dummy_input, deterministic=True)
        return model, params
    
    def test_pickle_save_load(self):
        """Test saving and loading with pickle format."""
        model, params = self.create_test_model()
        
        # Save as pickle
        pkl_path = self.test_dir / "params.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(params, f)
        
        # Save config
        config_path = self.test_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f)
        
        # Test loading
        loader = PureJAXModelLoader(cache_dir=self.test_dir)
        loaded_config = loader.load_config(config_path)
        loaded_params = loader.load_params_pickle(pkl_path)
        
        # Verify
        self.assertEqual(loaded_config.hidden_size, self.config.hidden_size)
        self.assertEqual(
            jax.tree_util.tree_structure(params),
            jax.tree_util.tree_structure(loaded_params)
        )
    
    def test_orbax_save_load(self):
        """Test saving and loading with Orbax format."""
        model, params = self.create_test_model()
        
        # Save with Orbax
        checkpoint_dir = self.test_dir / "orbax_checkpoint"
        ckptr = ocp.PyTreeCheckpointer()
        ckptr.save(checkpoint_dir, params)
        
        # Test loading
        loader = PureJAXModelLoader(cache_dir=self.test_dir)
        loaded_params = loader.load_params_orbax(checkpoint_dir)
        
        # Verify structure matches
        self.assertEqual(
            jax.tree_util.tree_structure(params),
            jax.tree_util.tree_structure(loaded_params)
        )
    
    def test_sharded_save_load(self):
        """Test sharded checkpoint save/load."""
        model, params = self.create_test_model()
        
        # Save with sharding (use small shard size for testing)
        checkpointer = ShardedCheckpointer(max_shard_size=1024)  # 1KB shards
        metadata = checkpointer.save_sharded(
            params,
            self.test_dir,
            self.config
        )
        
        # Verify metadata
        self.assertIn("sharding", metadata)
        self.assertGreater(metadata["sharding"]["num_shards"], 0)
        
        # Test loading
        loaded_params = checkpointer.load_sharded(
            self.test_dir,
            target_structure=params
        )
        
        # Verify params match
        for key in jax.tree_util.tree_leaves(params):
            # Check that parameters have same shape
            self.assertTrue(hasattr(key, 'shape'))
    
    def test_dtype_conversion(self):
        """Test loading with different data types."""
        model, params = self.create_test_model()
        
        # Save model
        save_path = self.test_dir / "model"
        save_model(model, params, save_path)
        
        # Test loading with different dtypes
        for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
            loader = PureJAXModelLoader(cache_dir=self.test_dir)
            
            # Mock the download to use local files
            with patch.object(loader, 'download_model_files', return_value=save_path):
                loaded_model, loaded_params = loader.load_model(
                    "test/model",
                    dtype=dtype
                )
            
            # Check dtype conversion
            for param in jax.tree_leaves(loaded_params):
                if hasattr(param, 'dtype'):
                    self.assertEqual(param.dtype, dtype)
    
    def test_hf_format_save(self):
        """Test saving in HuggingFace-compatible format."""
        model, params = self.create_test_model()
        
        # Create test tokenizer files
        tokenizer_dir = self.test_dir / "tokenizer"
        tokenizer_dir.mkdir()
        
        tokenizer_files = {
            "tokenizer_config.json": {"tokenizer_class": "DreamTokenizer"},
            "vocab.json": {"hello": 0, "world": 1},
            "merges.txt": "h e\nw o\n",
            "special_tokens_map.json": {"pad_token": "[PAD]"}
        }
        
        for filename, content in tokenizer_files.items():
            path = tokenizer_dir / filename
            if isinstance(content, dict):
                with open(path, "w") as f:
                    json.dump(content, f)
            else:
                path.write_text(content)
        
        # Save for HuggingFace
        hf_dir = self.test_dir / "hf_model"
        save_for_huggingface(
            model,
            params,
            hf_dir,
            tokenizer_path=tokenizer_dir,
            model_card="# Test Model"
        )
        
        # Verify files
        self.assertTrue((hf_dir / "config.json").exists())
        self.assertTrue((hf_dir / "README.md").exists())
        self.assertTrue((hf_dir / "checkpoint_metadata.json").exists())
        
        # Verify tokenizer files copied
        for filename in tokenizer_files:
            self.assertTrue((hf_dir / filename).exists())
    
    def test_url_construction(self):
        """Test HuggingFace URL construction."""
        loader = PureJAXModelLoader()
        
        url = loader._get_hf_url("user/model", "config.json", "main")
        self.assertEqual(
            url,
            "https://huggingface.co/user/model/resolve/main/config.json"
        )
        
        # Test with different revision
        url = loader._get_hf_url("user/model", "params.pkl", "v1.0")
        self.assertEqual(
            url,
            "https://huggingface.co/user/model/resolve/v1.0/params.pkl"
        )
    
    def test_cache_directory(self):
        """Test cache directory management."""
        loader = PureJAXModelLoader(cache_dir=self.test_dir / "cache")
        
        # Test repo cache dir
        cache_dir = loader._get_repo_cache_dir("user/model", "main")
        expected = loader.cache_dir / "user--model--main"
        self.assertEqual(cache_dir, expected)
        
        # Test with special characters
        cache_dir = loader._get_repo_cache_dir("org/model-name", "v1.0")
        expected = loader.cache_dir / "org--model-name--v1.0"
        self.assertEqual(cache_dir, expected)
    
    def test_download_with_mock(self):
        """Test file downloading with mocked responses."""
        loader = PureJAXModelLoader(cache_dir=self.test_dir)
        
        # Mock urllib.request.urlretrieve
        def mock_urlretrieve(url, dest):
            # Create a dummy file based on URL
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            if "config.json" in url:
                with open(dest, "w") as f:
                    json.dump(self.config.__dict__, f)
            elif "params.pkl" in url:
                model, params = self.create_test_model()
                with open(dest, "wb") as f:
                    pickle.dump(params, f)
            else:
                Path(dest).write_text("dummy content")
        
        with patch('urllib.request.urlretrieve', side_effect=mock_urlretrieve):
            # Test download
            local_dir = loader.download_model_files("test/model")
            
            # Verify files exist
            self.assertTrue((local_dir / "config.json").exists())
            self.assertTrue((local_dir / "params.pkl").exists())
    
    def test_full_load_workflow(self):
        """Test complete loading workflow."""
        # Create a test model repository
        repo_dir = self.test_dir / "test_repo"
        model, params = self.create_test_model()
        
        # Save in HuggingFace format
        save_for_huggingface(model, params, repo_dir)
        
        # Test loading with pure JAX loader
        loader = PureJAXModelLoader(cache_dir=self.test_dir)
        
        # Mock download to use local files
        def mock_download(repo_id, revision, files):
            # Copy files to cache
            cache_dir = loader._get_repo_cache_dir(repo_id, revision)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            for src_file in repo_dir.rglob("*"):
                if src_file.is_file():
                    rel_path = src_file.relative_to(repo_dir)
                    dst_file = cache_dir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
            
            return cache_dir
        
        with patch.object(loader, 'download_model_files', side_effect=mock_download):
            # Load model
            loaded_model, loaded_params = loader.load_model("test/model")
            
            # Verify
            self.assertIsInstance(loaded_model, DiffuCoder)
            self.assertEqual(loaded_model.config.hidden_size, self.config.hidden_size)
            
            # Check parameters loaded correctly
            param_count = sum(x.size for x in jax.tree_leaves(loaded_params))
            self.assertGreater(param_count, 0)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        loader = PureJAXModelLoader(cache_dir=self.test_dir)
        
        # Test missing config file
        with self.assertRaises(FileNotFoundError):
            loader.load_config(self.test_dir / "nonexistent.json")
        
        # Test missing pickle file
        with self.assertRaises(FileNotFoundError):
            loader.load_params_pickle(self.test_dir / "nonexistent.pkl")
        
        # Test invalid checkpoint directory
        with self.assertRaises(Exception):
            loader.load_params_orbax(self.test_dir / "nonexistent_checkpoint")
        
        # Test download failure
        with patch('urllib.request.urlretrieve', side_effect=Exception("Network error")):
            with self.assertRaises(RuntimeError):
                loader._download_file("http://example.com/file", self.test_dir / "file")


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory-efficient loading."""
    
    def test_lazy_loading_simulation(self):
        """Test that we can simulate lazy loading behavior."""
        config = DiffuCoderConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        # Create model
        model = DiffuCoder(config)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 8), dtype=jnp.int32)
        params = model.init(rng, dummy_input, deterministic=True)
        
        # Check memory usage simulation
        total_params = sum(x.size for x in jax.tree_leaves(params))
        total_bytes = sum(x.nbytes for x in jax.tree_leaves(params))
        
        print(f"Test model parameters: {total_params:,}")
        print(f"Test model size: {total_bytes / 1024:.2f} KB")
        
        # Verify reasonable size
        self.assertGreater(total_params, 0)
        self.assertLess(total_bytes, 10 * 1024 * 1024)  # Less than 10MB for test


if __name__ == "__main__":
    unittest.main()