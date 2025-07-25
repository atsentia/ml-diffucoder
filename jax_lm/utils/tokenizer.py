"""Tokenizer utilities for DiffuCoder."""

from pathlib import Path
from typing import List, Union, Optional

from transformers import AutoTokenizer


def load_tokenizer(
    tokenizer_path: Union[str, Path],
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """Load tokenizer from path or HuggingFace model ID.
    
    Args:
        tokenizer_path: Path to tokenizer or HF model ID
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=trust_remote_code,
    )
    
    # Add special tokens if needed
    special_tokens = {}
    
    # Add mask token if not present
    if tokenizer.mask_token is None:
        special_tokens["mask_token"] = "<|mask|>"
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<|pad|>"
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer


def prepare_input_ids(
    tokenizer: AutoTokenizer,
    texts: Union[str, List[str]],
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "np",
) -> dict:
    """Prepare input IDs for model input.
    
    Args:
        tokenizer: Tokenizer instance
        texts: Input text(s)
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        return_tensors: Return type ("np" for numpy, "jax" for JAX)
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    import jax.numpy as jnp
    
    # Tokenize
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="np",
    )
    
    # Convert to JAX if requested
    if return_tensors == "jax":
        encoded = {k: jnp.array(v) for k, v in encoded.items()}
    
    return encoded


def decode_sequences(
    tokenizer: AutoTokenizer,
    sequences: Union[List[int], List[List[int]]],
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> Union[str, List[str]]:
    """Decode token sequences back to text.
    
    Args:
        tokenizer: Tokenizer instance
        sequences: Token sequences to decode
        skip_special_tokens: Whether to skip special tokens
        clean_up_tokenization_spaces: Whether to clean up spaces
        
    Returns:
        Decoded text(s)
    """
    # Handle JAX arrays
    if hasattr(sequences, "tolist"):
        sequences = sequences.tolist()
    
    # Check if single sequence or batch
    if isinstance(sequences[0], int):
        # Single sequence
        return tokenizer.decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    else:
        # Batch of sequences
        return tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )