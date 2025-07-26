# Makefile for jax-diffucoder development

.PHONY: help install install-dev install-gpu install-tpu test lint format clean build upload docs

help:
	@echo "JAX-DiffuCoder Development Commands"
	@echo "==================================="
	@echo "install       - Install package with uv"
	@echo "install-dev   - Install with development dependencies"
	@echo "install-gpu   - Install with GPU support"
	@echo "install-tpu   - Install with TPU support"
	@echo "test          - Run all tests"
	@echo "test-fast     - Run fast tests only"
	@echo "lint          - Run linters"
	@echo "format        - Format code"
	@echo "clean         - Clean build artifacts"
	@echo "build         - Build package"
	@echo "upload-test   - Upload to TestPyPI"
	@echo "upload        - Upload to PyPI"

# Check if uv is installed
check-uv:
	@which uv > /dev/null || (echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)

install: check-uv
	uv pip install -e .

install-dev: check-uv
	uv pip install -e ".[dev,tokenizer]"
	uv pip install pre-commit
	pre-commit install

install-gpu: check-uv
	uv pip install -e ".[gpu,tokenizer]" \
		--extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

install-tpu: check-uv
	uv pip install -e ".[tpu,tokenizer]" \
		--extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html

test:
	python -m pytest tests/ -v --tb=short

test-fast:
	python -m pytest tests/ -v --tb=short -m "not slow"

test-loading:
	python -m pytest tests/test_model_loading.py -v -s

test-integration:
	python jax_lm/examples/test_huggingface_loading.py

lint:
	flake8 jax_lm/ --max-line-length=88 --extend-ignore=E203,W503
	mypy jax_lm/ --ignore-missing-imports
	black --check jax_lm/
	isort --check-only jax_lm/

format:
	black jax_lm/ tests/
	isort jax_lm/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .uv_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv pip install build
	python -m build

# Upload to TestPyPI for testing
upload-test: build
	uv pip install twine
	python -m twine upload --repository testpypi dist/*

# Upload to PyPI (requires authentication)
upload: build
	uv pip install twine
	python -m twine upload dist/*

# Benchmark commands
benchmark-cpu:
	python jax_lm/benchmarks/cpu_benchmark.py

benchmark-gpu:
	python jax_lm/benchmarks/hardware_benchmark.py --backend gpu

benchmark-tpu:
	python jax_lm/benchmarks/hardware_benchmark.py --backend tpu

# HuggingFace commands
hf-upload:
	python jax_lm/scripts/upload_to_huggingface_orbax.py \
		--model-path ./models/dream-jax \
		--repo-id $(REPO_ID) \
		--token $(HF_TOKEN)

hf-test-load:
	python jax_lm/examples/test_huggingface_loading.py

# Documentation
docs:
	@echo "Generating API documentation..."
	python -m pydoc -w jax_lm
	@echo "Documentation generated in current directory"

# Development setup
setup-dev: install-dev
	@echo "Setting up git hooks..."
	cp .github/hooks/* .git/hooks/ 2>/dev/null || true
	@echo "Development environment ready!"

# Quick checks before committing
pre-commit: format lint test-fast
	@echo "Pre-commit checks passed!"