# Loading HuggingFace Tokenizers

This document explains how to use the `load_from_huggingface` function to import vocabulary and merges from HuggingFace's tokenizer.json files.

## Overview

The `load_from_huggingface` method allows you to load pre-trained tokenizers from HuggingFace's format into the momo-tokenizer. This is useful when you want to:

- Use existing pre-trained tokenizers (GPT-2, BERT, RoBERTa, etc.)
- Convert HuggingFace tokenizers to the momo-tokenizer format
- Extend or modify existing tokenizers

## Usage

```rust
use momo_tokenizer::Tokenizer;

// Create a new tokenizer instance
let mut tokenizer = Tokenizer::new(None);

// Load from a local file
tokenizer.load_from_huggingface("path/to/tokenizer.json")?;

// Or load directly from a URL
tokenizer.load_from_huggingface("https://huggingface.co/gpt2/raw/main/tokenizer.json")?;

// Now you can use the tokenizer normally
let tokens = tokenizer.encode("Hello, world!");
let decoded = tokenizer.decode(&tokens);
```

## Loading HuggingFace Tokenizers

You can load tokenizers directly from URLs or download them locally first. The loader supports both methods:

### Direct URL Loading (Recommended)

```rust
// Load GPT-2 directly from HuggingFace
tokenizer.load_from_huggingface("https://huggingface.co/gpt2/raw/main/tokenizer.json")?;

// Load RoBERTa
tokenizer.load_from_huggingface("https://huggingface.co/roberta-base/raw/main/tokenizer.json")?;
```

### Local File Loading

You can also download tokenizer.json files locally first:

### GPT-2

```bash
wget https://huggingface.co/gpt2/raw/main/tokenizer.json -O gpt2_tokenizer.json
```

### BERT (Base Uncased)

```bash
wget https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json -O bert_tokenizer.json
```

### RoBERTa (Base)

```bash
wget https://huggingface.co/roberta-base/raw/main/tokenizer.json -O roberta_tokenizer.json
```

## What Gets Loaded

The `load_from_huggingface` function imports:

1. **Vocabulary**: The mapping from token strings to token IDs
2. **Merges**: The BPE merge rules that define how tokens are combined
3. **Special Tokens**: Any special tokens defined in the tokenizer (e.g., [CLS], [SEP], <|endoftext|>)

## Technical Details

### Byte-Level BPE Encoding

HuggingFace uses a byte-level BPE encoding scheme where certain bytes are mapped to Unicode characters to ensure all text is valid UTF-8. The loader handles this conversion automatically by:

1. Building a byte-to-Unicode mapping compatible with HuggingFace's scheme
2. Decoding token strings back to their original byte sequences
3. Reconstructing the merge hierarchy

### Compatibility Notes

- **BPE Tokenizers** (✅ Fully Supported): GPT-2, GPT-Neo, RoBERTa, DialoGPT, CodeGen, gpt-oss-120b
- **WordPiece Tokenizers** (❌ Not Supported): BERT, DistilBERT, ELECTRA
- **Unigram/SentencePiece** (❌ Not Supported): T5, XLNet, ALBERT, mBART
- **Other Types** (❌ Not Supported): Byte-Pair tokenizers that use different encoding schemes

The loader will check the tokenizer type and provide a clear error message if the tokenizer is not supported.

#### Merge Format Support

The loader automatically handles different merge formats used by various tokenizers:

- **String format** (`"token1 token2"`): Used by GPT-2, RoBERTa, and most BPE tokenizers
- **Array format** (`["token1", "token2"]`): Used by gpt-oss-120b and some newer models

Both formats are detected and processed automatically, ensuring broad compatibility with HuggingFace tokenizers.

## Example: Complete Workflow

```rust
use momo_tokenizer::Tokenizer;

fn main() -> std::io::Result<()> {
    // Create tokenizer
    let mut tokenizer = Tokenizer::new(None);

    // Load from URL
    tokenizer.load_from_huggingface("https://huggingface.co/gpt2/raw/main/tokenizer.json")?;

    // Or load from local file
    // tokenizer.load_from_huggingface("gpt2_tokenizer.json")?;

    // Display statistics
    tokenizer.stats();

    // Test encoding/decoding
    let text = "Hello, world! This is a test of the tokenizer.";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);

    println!("Original: {}", text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: {}", decoded);
    println!("Lossless: {}", text == decoded);

    // Save in momo-tokenizer format
    tokenizer.save("gpt2_converted")?;

    Ok(())
}
```

## URL Formats

When loading from HuggingFace, use the raw content URL format:

- ✅ Correct: `https://huggingface.co/gpt2/raw/main/tokenizer.json`
- ❌ Wrong: `https://huggingface.co/gpt2/blob/main/tokenizer.json` (this is the web view)

To get the correct URL:

1. Navigate to the tokenizer.json file on HuggingFace
2. Click the "download" or "raw" button
3. Copy that URL

## Troubleshooting

### Network Errors

If loading from URL fails, check:

- Internet connection
- URL is accessible (try opening in browser)
- URL points to raw JSON content, not HTML page

### File Not Found

For local files, ensure the path is correct and the file exists.

### Invalid JSON

Make sure you're using a valid HuggingFace tokenizer.json file. The file should contain:

- A "model" section with:
    - "type": "BPE" (only BPE is currently supported)
    - "vocab": object mapping tokens to IDs
    - "merges": array of merge rules
- Optionally, an "added_tokens" section

### Unsupported Tokenizer Type

If you get an "Unsupported tokenizer type" error, the tokenizer uses a different algorithm than BPE. Currently only BPE tokenizers are supported.

### Encoding/Decoding Mismatches

Some tokenizers may use special preprocessing that isn't captured in the tokenizer.json file. The loader focuses on the core BPE algorithm and may not reproduce all preprocessing steps.

### Performance

Loading large tokenizers (50k+ tokens) may take a few seconds. The loaded tokenizer is optimized for fast encoding/decoding after the initial load.
