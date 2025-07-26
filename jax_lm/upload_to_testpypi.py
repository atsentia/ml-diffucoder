#!/usr/bin/env python3
"""Upload jax-diffucoder to TestPyPI using token from .env file."""

import os
import subprocess
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                if key == "PYPI_TOKEN":
                    os.environ[key] = value.strip('"')

# Check if token is loaded
if "PYPI_TOKEN" not in os.environ:
    print("❌ PYPI_TOKEN not found in .env file")
    exit(1)

print("📦 Uploading jax-diffucoder to TestPyPI...")
print("=" * 50)

# Run twine upload
cmd = [
    "python", "-m", "twine", "upload",
    "--repository", "testpypi",
    "--username", "__token__",
    "--password", os.environ["PYPI_TOKEN"],
    "dist/*"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Upload successful!")
        print("\n📋 View your package at:")
        print("   https://test.pypi.org/project/jax-diffucoder/")
        print("\n🚀 Install with:")
        print("   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jax-diffucoder")
    else:
        print("❌ Upload failed!")
        print("Error:", result.stderr)
        
except Exception as e:
    print(f"❌ Error: {e}")