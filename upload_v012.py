#!/usr/bin/env python3
"""Upload jax-diffucoder v0.1.2 to TestPyPI."""

import os
import subprocess
from pathlib import Path

# Load token from jax_lm/.env
env_path = Path(__file__).parent / "jax_lm" / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                if key == "PYPI_TOKEN":
                    os.environ[key] = value.strip('"')

if "PYPI_TOKEN" not in os.environ:
    print("❌ PYPI_TOKEN not found")
    exit(1)

print("📦 Uploading jax-diffucoder v0.1.2 to TestPyPI...")

cmd = [
    "python", "-m", "twine", "upload",
    "--repository", "testpypi",
    "--username", "__token__",
    "--password", os.environ["PYPI_TOKEN"],
    "jax_diffucoder_pkg/dist/*0.1.2*"
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Upload successful!")
    print("\n🚀 Test in Colab with:")
    print("   !pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jax-diffucoder==0.1.2")
    print("\n📝 Then import with:")
    print("   import jax_lm")
    print("   print(f'Version: {jax_lm.__version__}')")
else:
    print("❌ Upload failed!")
    print(result.stderr)