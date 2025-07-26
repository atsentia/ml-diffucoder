"""Pure JAX model loading utilities - no HuggingFace dependencies.

This module provides loading functionality that only depends on JAX/Flax,
treating HuggingFace purely as a file storage service.
"""

import json
import pickle
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import tempfile
import shutil

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig


class PureJAXModelLoader:
    """Load JAX models without HuggingFace dependencies."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the loader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "jax_diffucoder"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_hf_url(self, repo_id: str, filename: str, revision: str = "main") -> str:
        """Construct HuggingFace download URL."""
        base_url = "https://huggingface.co"
        return f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    
    def _download_file(self, url: str, dest_path: Path, desc: str = "Downloading"):
        """Download a file with progress bar."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download to temp file first
        temp_path = dest_path.with_suffix('.tmp')
        
        try:
            print(f"{desc}: {url.split('/')[-1]}")
            urllib.request.urlretrieve(url, temp_path)
            
            # Move to final location
            shutil.move(str(temp_path), str(dest_path))
            return dest_path
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to download {url}: {e}")
    
    def _get_repo_cache_dir(self, repo_id: str, revision: str = "main") -> Path:
        """Get cache directory for a repository."""
        # Create a safe directory name from repo_id
        safe_name = repo_id.replace("/", "--")
        return self.cache_dir / f"{safe_name}--{revision}"
    
    def download_model_files(
        self,
        repo_id: str,
        revision: str = "main",
        files: Optional[List[str]] = None
    ) -> Path:
        """Download model files from HuggingFace.
        
        Args:
            repo_id: Repository ID (e.g., "atsentia/DiffuCoder-7B-JAX")
            revision: Git revision (branch, tag, commit)
            files: Specific files to download (None = all standard files)
            
        Returns:
            Path to local cache directory
        """
        cache_dir = self._get_repo_cache_dir(repo_id, revision)
        
        # Default files to download
        if files is None:
            files = [
                "config.json",
                "params.pkl",  # Legacy format
                "checkpoint_metadata.json",  # Orbax metadata
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            ]
            
            # Also check for Orbax checkpoint directory
            orbax_files = [
                "orbax_checkpoint/checkpoint",
                "orbax_checkpoint/commit_success.txt",
            ]
            files.extend(orbax_files)
        
        # Download each file
        downloaded = []
        for filename in files:
            dest_path = cache_dir / filename
            
            # Skip if already cached
            if dest_path.exists():
                print(f"Using cached: {filename}")
                downloaded.append(filename)
                continue
            
            # Try to download
            url = self._get_hf_url(repo_id, filename, revision)
            try:
                self._download_file(url, dest_path)
                downloaded.append(filename)
            except Exception as e:
                # Some files might not exist (e.g., orbax checkpoint)
                print(f"Skipping {filename}: {e}")
        
        if not downloaded:
            raise RuntimeError(f"No files downloaded from {repo_id}")
        
        return cache_dir
    
    def load_config(self, config_path: Path) -> DiffuCoderConfig:
        """Load model configuration."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Remove non-config fields
        config_dict.pop("model_type", None)
        config_dict.pop("architectures", None)
        config_dict.pop("framework", None)
        config_dict.pop("checkpoint_format", None)
        
        return DiffuCoderConfig(**config_dict)
    
    def load_params_pickle(self, pkl_path: Path) -> Dict[str, Any]:
        """Load parameters from pickle file."""
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    
    def load_params_orbax(
        self,
        checkpoint_dir: Path,
        target_structure: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load parameters from Orbax checkpoint."""
        ckptr = ocp.PyTreeCheckpointer()
        
        if target_structure:
            return ckptr.restore(checkpoint_dir, target=target_structure)
        else:
            return ckptr.restore(checkpoint_dir)
    
    def load_model(
        self,
        repo_id: str,
        revision: str = "main",
        dtype: Any = jnp.float32,
        force_download: bool = False
    ) -> Tuple[DiffuCoder, Dict[str, Any]]:
        """Load a model from HuggingFace repository.
        
        Args:
            repo_id: Repository ID (e.g., "atsentia/DiffuCoder-7B-JAX")
            revision: Git revision
            dtype: Data type for parameters
            force_download: Force re-download even if cached
            
        Returns:
            Tuple of (model, params)
        """
        # Clear cache if forced
        if force_download:
            cache_dir = self._get_repo_cache_dir(repo_id, revision)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        
        # Download files
        print(f"Loading model from {repo_id}...")
        local_dir = self.download_model_files(repo_id, revision)
        
        # Load config
        config = self.load_config(local_dir / "config.json")
        
        # Initialize model
        model = DiffuCoder(config, dtype=dtype)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
        init_params = model.init(rng, dummy_input, deterministic=True)
        
        # Load parameters - try different formats
        params = None
        
        # 1. Try pickle format
        pkl_path = local_dir / "params.pkl"
        if pkl_path.exists():
            print("Loading from pickle format...")
            params = self.load_params_pickle(pkl_path)
        
        # 2. Try Orbax format
        elif (local_dir / "orbax_checkpoint").exists():
            print("Loading from Orbax format...")
            params = self.load_params_orbax(
                local_dir / "orbax_checkpoint",
                init_params
            )
        
        # 3. Try sharded Orbax format  
        elif (local_dir / "checkpoint_metadata.json").exists():
            print("Loading from sharded Orbax format...")
            from jax_lm.utils.orbax_sharding import ShardedCheckpointer
            checkpointer = ShardedCheckpointer()
            params = checkpointer.load_sharded(local_dir, init_params, dtype)
        
        else:
            raise ValueError(f"No valid checkpoint found in {local_dir}")
        
        # Convert dtype if needed
        if dtype != jnp.float32:
            params = jax.tree_map(lambda x: x.astype(dtype), params)
        
        print(f"✅ Model loaded successfully")
        param_count = sum(x.size for x in jax.tree_leaves(params))
        print(f"   Parameters: {param_count:,}")
        
        return model, params


# Convenience functions
_default_loader = None

def get_default_loader() -> PureJAXModelLoader:
    """Get the default model loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PureJAXModelLoader()
    return _default_loader


def load_model(
    repo_id: str,
    revision: str = "main", 
    dtype: Any = jnp.float32,
    cache_dir: Optional[Path] = None,
    force_download: bool = False
) -> Tuple[DiffuCoder, Dict[str, Any]]:
    """Load a JAX model from HuggingFace.
    
    This is a convenience function that uses the default loader.
    
    Args:
        repo_id: HuggingFace repository ID
        revision: Git revision
        dtype: Parameter data type
        cache_dir: Custom cache directory
        force_download: Force re-download
        
    Returns:
        Tuple of (model, params)
    """
    if cache_dir:
        loader = PureJAXModelLoader(cache_dir)
    else:
        loader = get_default_loader()
    
    return loader.load_model(repo_id, revision, dtype, force_download)


def load_tokenizer(repo_id: str, revision: str = "main") -> Any:
    """Load tokenizer from HuggingFace repository.
    
    Args:
        repo_id: Repository ID
        revision: Git revision
        
    Returns:
        Tokenizer instance
    """
    loader = get_default_loader()
    local_dir = loader.download_model_files(repo_id, revision)
    
    # Import tokenizer utilities
    from jax_lm.utils.tokenizer import DreamTokenizer
    
    # Load tokenizer from local files
    return DreamTokenizer.from_pretrained(str(local_dir))