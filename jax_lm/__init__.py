"""
jax-diffucoder: High-performance JAX/Flax implementation of DiffuCoder
"""

__version__ = "0.1.0"
__author__ = "ML-DiffuCoder Contributors"

# Import model components
from jax_lm.models.dream import DreamConfig, DreamForCausalLM
from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig

# Import generation functions
from jax_lm.generate_diffusion import (
    diffusion_generate,
    stream_diffusion_generate,
)

# Import utilities
from jax_lm.utils.model_utils import (
    load_model,
    save_model,
    convert_pytorch_checkpoint,
)
from jax_lm.utils.tokenizer import load_tokenizer

__all__ = [
    "DiffuCoder",
    "DiffuCoderConfig",
    "diffusion_generate",
    "stream_diffusion_generate",
    "load_model",
    "save_model",
    "convert_pytorch_checkpoint",
]