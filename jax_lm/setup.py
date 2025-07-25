"""Setup configuration for jax_lm package."""

from setuptools import setup, find_packages

setup(
    name="jax_lm",
    version="0.1.0",
    author="JAX DiffuCoder Contributors",
    description="JAX/Flax implementation of DiffuCoder for TPU-optimized code generation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.25",
        "jaxlib>=0.4.25",
        "flax>=0.8.0",
        "optax>=0.1.9",
        "orbax-checkpoint>=0.5.0",
        "einops>=0.8.0",
        "transformers>=4.36.0",
        "sentencepiece>=0.1.99",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "ml_collections",
        "chex>=0.1.85",
        "tensorflow-datasets>=4.9.0",  # For data loading utilities
    ],
    extras_require={
        "tpu": ["jax[tpu]>=0.4.25"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-xdist",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)