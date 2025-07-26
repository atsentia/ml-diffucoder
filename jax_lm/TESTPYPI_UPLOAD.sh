#!/bin/bash
# TestPyPI Upload Script

echo "=== TestPyPI Upload Guide ==="
echo
echo "1. First, create a TestPyPI account at:"
echo "   https://test.pypi.org/account/register/"
echo
echo "2. Create an API token at:"
echo "   https://test.pypi.org/manage/account/"
echo "   -> 'Add API token' -> Name: 'jax-diffucoder-test' -> Scope: 'Entire account'"
echo
echo "3. The token will start with 'pypi-'. Copy it!"
echo
echo "4. Run this command to upload:"
echo "   python -m twine upload --repository testpypi dist/*"
echo
echo "5. When prompted:"
echo "   - Username: __token__"
echo "   - Password: [paste your token]"
echo
echo "6. After upload, view your package at:"
echo "   https://test.pypi.org/project/jax-diffucoder/"
echo
echo "7. Test installation in Colab with:"
echo "   !pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jax-diffucoder"
echo
echo "Ready to upload? Press Enter to continue or Ctrl+C to cancel..."
read

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*