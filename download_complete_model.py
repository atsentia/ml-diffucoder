#!/usr/bin/env python3
"""
Download complete DiffuCoder model weights for benchmarking.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def download_diffucoder_model():
    """Download the complete DiffuCoder-7B model."""
    
    model_id = "apple/DiffuCoder-7B-Base"
    output_dir = Path("./models/diffucoder-7b-complete")
    
    print(f"🚀 Downloading {model_id} to {output_dir}")
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model files
        print("📥 Downloading model files...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            cache_dir=str(output_dir.parent / "cache"),
            resume_download=True,
        )
        
        print("✅ Model download completed!")
        
        # Verify the download by loading the model
        print("🔍 Verifying model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
            
            # Load model with torch_dtype to avoid warnings
            model = AutoModelForCausalLM.from_pretrained(
                str(output_dir),
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Load on CPU to avoid memory issues during verification
                trust_remote_code=True,
            )
            
            print(f"✅ Model verification successful!")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Model dtype: {model.dtype}")
            print(f"   Vocab size: {tokenizer.vocab_size}")
            print(f"   Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
            
        except Exception as e:
            print(f"⚠️  Model verification failed: {e}")
            print("Model downloaded but may have issues")
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "download_path": str(output_dir),
            "model_type": "DiffuCoder",
            "size": "7B",
            "downloaded": True,
        }
        
        with open(output_dir / "download_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(output_dir)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

def main():
    """Main function."""
    print("🔽 DiffuCoder Complete Model Downloader")
    print("=" * 50)
    
    try:
        model_path = download_diffucoder_model()
        print(f"\n✅ Success! Model downloaded to: {model_path}")
        print("\nNext steps:")
        print("1. Run benchmarks with: python run_complete_benchmark.py")
        print("2. Convert to JAX format if needed")
        
    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set HF cache directory
    os.environ.setdefault("HF_HOME", "./models/cache")
    main()