#!/usr/bin/env python3
"""Quick verification of sharded model files structure."""

import json
from pathlib import Path
import os
import sys

# Force CPU mode
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"


def verify_sharded_structure(model_dir: Path):
    """Verify the structure of sharded model files."""
    print(f"\nVerifying sharded model structure in: {model_dir}")
    print("=" * 60)
    
    if not model_dir.exists():
        print(f"❌ Directory not found: {model_dir}")
        return False
    
    # Expected files
    expected_files = {
        "config.json": "Model configuration",
        "checkpoint_metadata.json": "Sharding metadata",
        "README.md": "Model card",
        "loading_instructions.py": "Usage example",
    }
    
    # Tokenizer files (optional)
    tokenizer_files = {
        "tokenizer_config.json": "Tokenizer config",
        "vocab.json": "Vocabulary",
        "merges.txt": "BPE merges",
        "special_tokens_map.json": "Special tokens",
    }
    
    found_files = []
    missing_files = []
    
    # Check expected files
    print("\n📁 Required files:")
    for file_name, description in expected_files.items():
        file_path = model_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file_name:<30} ({size:>10,} bytes) - {description}")
            found_files.append(file_name)
        else:
            print(f"  ❌ {file_name:<30} - {description}")
            missing_files.append(file_name)
    
    # Check tokenizer files
    print("\n📝 Tokenizer files:")
    for file_name, description in tokenizer_files.items():
        file_path = model_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file_name:<30} ({size:>10,} bytes) - {description}")
            found_files.append(file_name)
        else:
            print(f"  ⚠️  {file_name:<30} - {description} (optional)")
    
    # Check Orbax checkpoint
    checkpoint_dir = model_dir / "orbax_checkpoint"
    if checkpoint_dir.exists():
        print(f"\n📦 Orbax checkpoint directory:")
        checkpoint_files = list(checkpoint_dir.rglob("*"))
        total_size = 0
        
        # Group files by type
        zarr_files = []
        metadata_files = []
        other_files = []
        
        for file_path in checkpoint_files:
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                rel_path = file_path.relative_to(checkpoint_dir)
                
                if str(rel_path).endswith('.zarr') or '.zarr/' in str(rel_path):
                    zarr_files.append((rel_path, size))
                elif 'metadata' in str(rel_path) or str(rel_path).endswith('.json'):
                    metadata_files.append((rel_path, size))
                else:
                    other_files.append((rel_path, size))
        
        # Display organized files
        if zarr_files:
            print("  📊 Zarr shards:")
            for path, size in sorted(zarr_files)[:10]:
                print(f"    - {str(path):<40} ({size / 1e6:>8.1f} MB)")
            if len(zarr_files) > 10:
                print(f"    ... and {len(zarr_files) - 10} more files")
        
        if metadata_files:
            print("  📋 Metadata files:")
            for path, size in sorted(metadata_files):
                print(f"    - {str(path):<40} ({size:>10,} bytes)")
        
        if other_files:
            print("  📄 Other files:")
            for path, size in sorted(other_files)[:5]:
                print(f"    - {str(path):<40} ({size / 1e6:>8.1f} MB)")
            if len(other_files) > 5:
                print(f"    ... and {len(other_files) - 5} more files")
        
        print(f"\n  Total checkpoint size: {total_size / 1e9:.2f} GB")
        print(f"  Total files: {len(checkpoint_files)}")
    else:
        print(f"\n❌ Orbax checkpoint directory not found")
        missing_files.append("orbax_checkpoint")
    
    # Load and display metadata
    metadata_path = model_dir / "checkpoint_metadata.json"
    if metadata_path.exists():
        print("\n📊 Checkpoint metadata:")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        for key, value in metadata.items():
            if key != "parameters":  # Skip large parameter lists
                print(f"  - {key}: {value}")
    
    # Summary
    print("\n📈 Summary:")
    print(f"  - Found files: {len(found_files)}")
    print(f"  - Missing required files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
    
    return len(missing_files) == 0


def main():
    """Main function."""
    print("🔍 JAX Sharded Model Verifier")
    
    # Parse arguments
    if len(sys.argv) > 1:
        model_dirs = [Path(arg) for arg in sys.argv[1:]]
    else:
        # Default directories
        model_dirs = [
            Path("models/dream-jax"),
            Path("models/dream-jax-sharded"),
        ]
    
    all_valid = True
    
    for model_dir in model_dirs:
        valid = verify_sharded_structure(model_dir)
        all_valid = all_valid and valid
    
    if all_valid:
        print("\n✅ All model directories have valid structure!")
        return 0
    else:
        print("\n❌ Some model directories have issues!")
        return 1


if __name__ == "__main__":
    sys.exit(main())