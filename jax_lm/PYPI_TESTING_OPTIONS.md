# PyPI Testing Options (Private/Test Uploads)

## 1. TestPyPI (Recommended for Testing)

TestPyPI is a separate instance of PyPI for testing. It's public but separate from the main index.

### Setup
1. Create account at https://test.pypi.org/account/register/
2. Get API token from https://test.pypi.org/manage/account/token/

### Upload
```bash
cd jax_lm
python -m twine upload --repository testpypi dist/*
```

### Install from TestPyPI
```bash
# In Colab or locally
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jax-diffucoder
```

Note: Use `--extra-index-url` because dependencies might not be on TestPyPI.

## 2. Private PyPI Server Options

### A. PyPI Cloud (Managed)
- https://pypicloud.com/
- Free tier available
- Easy setup, no maintenance

### B. Artifactory/Nexus
- Enterprise solutions
- More complex, better for organizations

### C. Simple HTTP Server (Quick & Dirty)
```bash
# In the dist directory
cd jax_lm/dist
python -m http.server 8000

# Install from private server
pip install --index-url http://localhost:8000 jax-diffucoder-0.1.0-py3-none-any.whl
```

### D. GitHub Releases (Semi-Private)
- Upload wheel to GitHub releases
- Install directly from GitHub:
```bash
pip install https://github.com/atsentia/ml-diffucoder/releases/download/v0.1.0/jax_diffucoder-0.1.0-py3-none-any.whl
```

## 3. Direct from Git (Current Option)

While testing, you can install directly from git:
```bash
# Install from specific branch/tag
pip install git+https://github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm

# Or for private repo with token
pip install git+https://<token>@github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm
```

## 4. Google Cloud Artifact Registry (For Colab)

If you have GCP access:
```bash
# Create Python repository
gcloud artifacts repositories create jax-diffucoder-test \
    --repository-format=python \
    --location=us-central1

# Configure pip
pip config set global.extra-index-url https://us-central1-python.pkg.dev/PROJECT_ID/jax-diffucoder-test/simple/

# Upload
twine upload --repository-url https://us-central1-python.pkg.dev/PROJECT_ID/jax-diffucoder-test/ dist/*
```

## Recommendation

For immediate testing:
1. **Use TestPyPI** - It's free, public but separate, and works well with Colab
2. **GitHub Releases** - If you want more control over who can access

For your use case, TestPyPI is probably the best option since:
- It's designed for exactly this purpose
- Works seamlessly with pip
- Allows full end-to-end testing
- Easy to promote to real PyPI once tested