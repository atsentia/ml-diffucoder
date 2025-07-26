# GitHub Authentication in Google Colab

## 1. Public Repositories (No Auth Needed)

For public repos, no authentication is required:

```python
# Works without any auth
!pip install git+https://github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm
```

## 2. Private Repositories

### Option A: Personal Access Token (PAT)

```python
# Method 1: Direct in URL (not recommended - token visible)
!pip install git+https://<YOUR_PAT>@github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm

# Method 2: Using environment variable (better)
import os
os.environ['GITHUB_TOKEN'] = 'your_pat_here'  # Or use Colab secrets
!pip install git+https://$GITHUB_TOKEN@github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm

# Method 3: Using getpass (interactive)
from getpass import getpass
token = getpass('Enter GitHub token: ')
!pip install git+https://{token}@github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm
```

### Option B: Colab Secrets (Recommended for Private Repos)

1. In Colab, click the 🔑 key icon in the left sidebar
2. Add a new secret named `GITHUB_TOKEN`
3. Use in notebook:

```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')
!pip install git+https://{token}@github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm
```

### Option C: SSH Keys (More Complex)

```python
# Generate SSH key
!ssh-keygen -t ed25519 -C "colab@example.com" -f /root/.ssh/id_ed25519 -N ""

# Display public key to add to GitHub
!cat /root/.ssh/id_ed25519.pub

# Configure git
!git config --global core.sshCommand "ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

# Clone private repo
!git clone git@github.com:atsentia/ml-diffucoder.git
```

## 3. GitHub API Access

For API calls (not package installation):

```python
import requests

# Public API (no auth, rate limited to 60 requests/hour)
response = requests.get('https://api.github.com/repos/atsentia/ml-diffucoder')

# Authenticated API (5000 requests/hour)
headers = {'Authorization': f'token {token}'}
response = requests.get('https://api.github.com/user/repos', headers=headers)
```

## 4. Mounting Google Drive for Credentials

```python
from google.colab import drive
drive.mount('/content/drive')

# Read token from a file in Drive
with open('/content/drive/MyDrive/github_token.txt', 'r') as f:
    token = f.read().strip()
```

## Current Situation

Since **ml-diffucoder is a public repository**, you don't need any authentication:

```python
# This works right now in any Colab notebook:
!pip install git+https://github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm

# Test it
import jax_lm
print(jax_lm.__version__)
```

## Security Best Practices

1. **Never hardcode tokens** in notebooks you share
2. **Use Colab Secrets** for private repos
3. **Revoke tokens** after use if needed
4. **Use fine-grained PATs** with minimal permissions
5. **Consider read-only tokens** for pip installs

## Creating a GitHub Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token"
3. For public repo pip install: No special permissions needed
4. For private repo pip install: Select `repo` scope
5. Copy the token (you won't see it again)

For your public repo, no auth is needed - anyone can install directly from GitHub!