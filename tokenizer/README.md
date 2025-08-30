# Momo Tokenizer - Rust Implementation

A Rust implementation of a Byte Pair Encoding (BPE) tokenizer, ported from the Python version in `tokenizer.py`.

## Features

- **Byte Pair Encoding (BPE)**: Implements the BPE algorithm for text tokenization
- **Two Training Methods**:
    - **Naive**: O(3mn) complexity, recalculates frequencies after each merge
    - **Optimized**: O(2mn) complexity, tracks frequency changes during merging
- **GPT-4 Compatible Pattern**: Uses a regex pattern similar to GPT-4's tokenization
- **Parallel Processing**: Utilizes Rayon for parallel token merging
- **Progress Bars**: Visual training progress using indicatif
- **Model Persistence**: Save and load trained tokenizers
- **Lossless Encoding**: Perfect reconstruction of original text from tokens

## Usage

### Building

```bash
cd momo-tokenizer
cargo build --release
```

### Running

```bash
cargo run --release
```

### As a Library

## Python Usage

```python
from momo_tokenizer import Tokenizer

# Create a tokenizer instance
tokenizer = Tokenizer()

# Load a HuggingFace tokenizer (from URL)
tokenizer.load_from_huggingface("https://huggingface.co/gpt2/raw/main/tokenizer.json")

# Or load from local file
# tokenizer.load_from_huggingface("path/to/tokenizer.json")

# Encode text to tokens
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")  # [15496, 995, 0]

# Decode tokens back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")  # "Hello, world!"

# Get tokenizer information
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Number of merges: {tokenizer.num_merges}")
print(f"Special tokens: {tokenizer.num_special_tokens}")
```

### Example: Using gpt-oss-120b tokenizer

```python
from momo_tokenizer import Tokenizer

# Load the gpt-oss-120b tokenizer
tokenizer = Tokenizer()
tokenizer.load_from_huggingface(
    "https://huggingface.co/openai/gpt-oss-120b/resolve/main/tokenizer.json"
)

# Tokenize some text
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(text)
print(f"Original: {text}")
print(f"Tokens: {tokens[:10]}... ({len(tokens)} total)")
print(f"Compression ratio: {len(text) / len(tokens):.2f}")
```

### Running the Example Script

```bash
python example_gpt_oss_tokenizer.py
```

This will demonstrate:

- Loading the gpt-oss-120b tokenizer from HuggingFace
- Encoding various types of text (emojis, multiple languages, code)
- Performance benchmarking
- Batch processing

## Rust Usage

```rust
use momo_tokenizer::Tokenizer;

// Create a new tokenizer
let mut tokenizer = Tokenizer::new(None);

// Load from HuggingFace
tokenizer.load_from_huggingface("https://huggingface.co/gpt2/raw/main/tokenizer.json")?;

// Encode and decode
let tokens = tokenizer.encode("Hello, world!");
let decoded = tokenizer.decode(&tokens);
```

## Supported Tokenizers

### ✅ Fully Supported (BPE)

- GPT-2 (all sizes)
- GPT-Neo / GPT-J
- RoBERTa
- DialoGPT
- CodeGen
- gpt-oss-120b

### ❌ Not Supported

- BERT (WordPiece)
- T5 (SentencePiece)
- XLNet (SentencePiece)
  println!("Decoded: {}", text);

// Save model
tokenizer.save("my_tokenizer").unwrap();

// Load model
let mut loaded_tokenizer = Tokenizer::new(None);
loaded_tokenizer.load("my_tokenizer.model").unwrap();

```

## Implementation Details

### Key Components

1. **Pattern Matching**: Uses regex to split text into logical chunks before tokenization
2. **Frequency Tracking**: Maintains pair frequencies for efficient merge selection
3. **Vocabulary Building**: Constructs token-to-bytes mapping from merges
4. **Parallel Processing**: Merges tokens across chunks in parallel for better performance

### Differences from Python Version

- **Regex Pattern**: Modified to remove lookahead assertions (not supported in Rust regex)
- **Parallel Processing**: Uses Rayon instead of Python's multiprocessing
- **Progress Bars**: Uses indicatif instead of tqdm
- **Error Handling**: Uses Result types for better error handling

### File Format

The tokenizer saves two files:
- `.model`: Contains the tokenizer configuration and merges (used for loading)
- `.vocab`: Human-readable vocabulary listing (for inspection only)

Model file format:
```

bpetokenizer v1
<regex_pattern>
<num_special_tokens>
<special_token> <token_id>
...
<merge_id1> <merge_id2> <result_token_id>
...

```

## Performance

The Rust implementation offers comparable compression ratios to the Python version with better performance:
- Typical compression ratio: 1.5-2.5x depending on text
- Training speed: Significantly faster than Python due to parallel processing
- Memory usage: More efficient due to Rust's ownership model

## Dependencies

- `regex`: For pattern matching and text chunking
- `indicatif`: For progress bars during training
- `rayon`: For parallel processing during merging

## License

Same as the parent project.
```
