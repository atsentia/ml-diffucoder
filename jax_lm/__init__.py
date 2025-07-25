"""JAX/Flax implementation of DiffuCoder."""

# Import the actual model classes (Dream is the model name)
from jax_lm.models.dream import DreamForCausalLM as DiffuCoder
from jax_lm.models.dream import DreamConfig as DiffuCoderConfig

# Generation functions
from jax_lm.generate_diffusion import (
    diffusion_generate,
    stream_diffusion_generate,
)

# Model utilities
from jax_lm.utils.model_utils import (
    load_model,
    save_model,
    convert_pytorch_checkpoint,
    load_config,
    save_config,
    count_parameters,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "DiffuCoder",
    "DiffuCoderConfig",
    # Generation
    "diffusion_generate",
    "stream_diffusion_generate",
    # Utilities
    "load_model",
    "save_model",
    "convert_pytorch_checkpoint",
    "load_config",
    "save_config",
    "count_parameters",
]