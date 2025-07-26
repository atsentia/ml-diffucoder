# TestPyPI Quick Start

## 1. Create TestPyPI Account
Go to: https://test.pypi.org/account/register/

## 2. Create API Token
1. Go to: https://test.pypi.org/manage/account/token/
2. Create token with scope "Entire account"
3. Copy the token (starts with `pypi-`)

## 3. Upload to TestPyPI

```bash
cd jax_lm

# Upload using token
python -m twine upload --repository testpypi dist/* --username __token__ --password <your-token>

# Or save token in ~/.pypirc for convenience
```

## 4. Test in Colab

Create a new Colab notebook and run:

```python
# Cell 1: Install from TestPyPI
!pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jax-diffucoder

# Cell 2: Test import
import jax_lm
print(f"jax-diffucoder version: {jax_lm.__version__}")

# Cell 3: Test basic functionality
from jax_lm import DiffuCoderConfig
config = DiffuCoderConfig()
print(f"Model config: hidden_size={config.hidden_size}")
```

## 5. View Package Page

Once uploaded, view at:
https://test.pypi.org/project/jax-diffucoder/

## Notes

- TestPyPI is completely separate from PyPI
- Packages on TestPyPI are public but won't show up in regular pip searches
- TestPyPI occasionally gets wiped, so don't rely on it for permanent storage
- Once tested, upload to real PyPI with: `python -m twine upload dist/*`

## Alternative: Direct GitHub Install

If you prefer not to use TestPyPI at all, users can install directly from GitHub:

```python
# In Colab
!pip install git+https://github.com/atsentia/ml-diffucoder.git@main#subdirectory=jax_lm
```