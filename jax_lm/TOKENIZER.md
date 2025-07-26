# DiffuCoder Tokenizer Guide

This guide covers the tokenizer implementation for DiffuCoder JAX/Flax models.

## Overview

DiffuCoder uses a custom BPE (Byte Pair Encoding) tokenizer with a vocabulary size of 151,643 tokens. The tokenizer is optimized for code generation and includes special tokens for various programming constructs.

## Installation

The tokenizer is included with the JAX implementation. Required files:
- `vocab.json` - Vocabulary mappings (151,643 entries)
- `merges.txt` - BPE merge rules
- `tokenizer_config.json` - Configuration
- `tokenization_dream.py` - Official implementation

## Usage

### Basic Usage

```python
from diffucoder_tokenizer import load_diffucoder_tokenizer

# Load the tokenizer
tokenizer = load_diffucoder_tokenizer("./models/diffucoder-7b-complete")

# Encode text
text = "def hello_world():\n    return 'Hello, World!'"
inputs = tokenizer(text, return_tensors="jax")
print(f"Input IDs: {inputs['input_ids']}")
print(f"Shape: {inputs['input_ids'].shape}")

# Decode tokens
decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
print(f"Decoded: {decoded}")
```

### Batch Processing

```python
# Tokenize multiple texts
texts = [
    "def fibonacci(n):",
    "class BinaryTree:",
    "print('Hello, World!')"
]

# Batch encode with padding
inputs = tokenizer(texts, return_tensors="jax", padding=True)
print(f"Batch shape: {inputs['input_ids'].shape}")
```

### Advanced Options

```python
# With truncation and max length
inputs = tokenizer(
    text,
    return_tensors="jax",
    padding="max_length",
    max_length=512,
    truncation=True
)

# Return numpy arrays instead of JAX
inputs_np = tokenizer(text, return_tensors="np")
```

## Special Tokens

The tokenizer includes several special tokens:

| Token | ID | Description |
|-------|-----|-------------|
| `<\|beginoftext\|>` | 151665 | Beginning of text |
| `<\|endoftext\|>` | 151643 | End of text |
| `<\|mask\|>` | 151666 | Mask token for diffusion |
| `<\|dlm_pad\|>` | 151667 | Padding token |
| `<\|fim_prefix\|>` | 151659 | Fill-in-middle prefix |
| `<\|fim_middle\|>` | 151660 | Fill-in-middle middle |
| `<\|fim_suffix\|>` | 151661 | Fill-in-middle suffix |

## Performance Characteristics

Based on our benchmarks on TPU:

- **Initial tokenization**: ~200ms (includes loading/caching)
- **Subsequent tokenization**: ~20-25ms
- **Single token decode**: <1ms
- **Batch decode (100 tokens)**: ~1ms

## Implementation Details

The tokenizer provides two implementations:

1. **Official DreamTokenizer**: Uses the HuggingFace transformers library
2. **SimpleDiffuCoderTokenizer**: Fallback pure-Python implementation

Both support:
- BPE tokenization
- Special token handling
- Batch processing
- JAX/NumPy tensor outputs

## Troubleshooting

### Common Issues

1. **Slow first tokenization**: This is normal due to initial loading. Subsequent calls are much faster.

2. **Memory usage**: The vocabulary is large (151k tokens). Ensure sufficient RAM.

3. **Special tokens in output**: Use `skip_special_tokens=True` when decoding.

### Example: Optimized Generation

For best performance with JAX, use fixed-length sequences:

```python
# Fixed-length tokenization to avoid JIT recompilation
MAX_SEQ_LEN = 512

inputs = tokenizer(
    prompt,
    return_tensors="jax",
    padding="max_length",
    max_length=MAX_SEQ_LEN,
    truncation=True
)
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest test_tokenizer.py -v
```

All 16 tests should pass, covering:
- Basic encoding/decoding
- Batch processing
- Special tokens
- Unicode handling
- Performance benchmarks