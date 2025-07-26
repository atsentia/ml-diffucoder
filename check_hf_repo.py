#!/usr/bin/env python3
"""Check HuggingFace repository status."""

import os
import requests
from pathlib import Path

# Load HF token
env_path = Path(__file__).parent / "jax_lm" / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                if key == "HF_TOKEN":
                    os.environ[key] = value.strip('"')

token = os.environ.get("HF_TOKEN")
if not token:
    print("❌ HF_TOKEN not found")
    exit(1)

# Check if repo exists
repo_id = "atsentia/DiffuCoder-7B-JAX"
headers = {"Authorization": f"Bearer {token}"}

# Check repo
response = requests.get(
    f"https://huggingface.co/api/models/{repo_id}",
    headers=headers
)

if response.status_code == 200:
    print(f"✅ Repository exists: https://huggingface.co/{repo_id}")
    repo_info = response.json()
    print(f"   Private: {repo_info.get('private', False)}")
    print(f"   Files: {len(repo_info.get('siblings', []))}")
elif response.status_code == 404:
    print(f"❌ Repository not found: {repo_id}")
    print("\nTo create it:")
    print("1. Go to https://huggingface.co/new")
    print("2. Name: DiffuCoder-7B-JAX")
    print("3. Make it public")
    print("4. Select 'Model' type")
else:
    print(f"❌ Error checking repo: {response.status_code}")
    print(response.text)

# Check permissions
print("\nChecking token permissions...")
whoami_response = requests.get(
    "https://huggingface.co/api/whoami",
    headers=headers
)

if whoami_response.status_code == 200:
    user_info = whoami_response.json()
    print(f"✅ Authenticated as: {user_info.get('name', 'unknown')}")
    
    # Check if user has write access
    orgs = user_info.get('orgs', [])
    if any(org['name'] == 'atsentia' for org in orgs):
        print("✅ You have access to atsentia organization")
    else:
        print("❌ No access to atsentia organization")
        print("   You may need to request access or use your personal namespace")
else:
    print(f"❌ Authentication failed: {whoami_response.status_code}")