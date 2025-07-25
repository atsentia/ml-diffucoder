#!/usr/bin/env python3
"""Download DiffuCoder weights from HuggingFace."""

import argparse
import os
from pathlib import Path
import json

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def download_diffucoder_weights(model_id: str, output_dir: str, revision: str = "main"):
    """Download DiffuCoder model weights from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID (e.g., "apple/DiffuCoder-7B-Instruct")
        output_dir: Directory to save the downloaded weights
        revision: Git revision to download (default: "main")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_id} to {output_path}")
    
    # Download model files
    cache_dir = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=output_path / "cache",
        local_dir=output_path / "pytorch_weights",
        ignore_patterns=["*.safetensors"],  # Download only .bin files for now
    )
    
    # Download tokenizer separately
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=output_path / "cache",
    )
    tokenizer.save_pretrained(output_path / "tokenizer")
    
    # Create metadata file
    metadata = {
        "model_id": model_id,
        "revision": revision,
        "download_path": str(output_path),
        "pytorch_weights_path": str(output_path / "pytorch_weights"),
        "tokenizer_path": str(output_path / "tokenizer"),
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Successfully downloaded {model_id}")
    print(f"   PyTorch weights: {output_path / 'pytorch_weights'}")
    print(f"   Tokenizer: {output_path / 'tokenizer'}")
    print(f"   Metadata: {output_path / 'metadata.json'}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download DiffuCoder weights from HuggingFace"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="apple/DiffuCoder-7B-Instruct",
        help="HuggingFace model ID to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/diffucoder-7b-instruct",
        help="Directory to save the downloaded weights",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision to download",
    )
    
    args = parser.parse_args()
    download_diffucoder_weights(args.model_id, args.output_dir, args.revision)


if __name__ == "__main__":
    main()