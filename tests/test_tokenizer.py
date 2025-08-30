#!/usr/bin/env python3
"""
Simple test script for the momo_tokenizer Python bindings
"""

import time
from tokenizer import Tokenizer

def main():
    print("=== MoMo Tokenizer Test ===\n")

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = Tokenizer()

    # Load the gpt-oss-120b tokenizer
    url = "https://huggingface.co/BEE-spoke-data/cl100k_base-mlm/resolve/main/tokenizer.json"
    print(f"Loading tokenizer from:\n{url}")

    start_time = time.time()
    try:
        tokenizer.load_from_huggingface(url)
        load_time = time.time() - start_time
        print(f"✓ Tokenizer loaded successfully in {load_time:.2f}s")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Show stats
    print(f"\nTokenizer Statistics:")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"  Number of merges: {tokenizer.num_merges:,}")
    print(f"  Special tokens: {tokenizer.num_special_tokens}")

    # Test strings
    test_strings = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "OpenAI's GPT models are powerful language models.",
        "Testing with numbers: 123, 456.789",
        "🚀 Unicode and emojis work too! 🎉",
    ]

    print("\n" + "-"*60)
    print("Tokenization Examples")
    print("-"*60)

    for text in test_strings:
        # Encode
        start_time = time.time()
        tokens = tokenizer.encode(text)
        encode_time = time.time() - start_time

        # Decode
        start_time = time.time()
        decoded = tokenizer.decode(tokens)
        decode_time = time.time() - start_time

        # Display results
        print(f"\nText: {text}")
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} ({len(tokens)} tokens)")
        print(f"Decoded: {decoded}")
        print(f"Match: {'✓' if text == decoded else '✗'}")
        print(f"Time: encode={encode_time*1000:.2f}ms, decode={decode_time*1000:.2f}ms")

        # Compression ratio
        bytes_count = len(text.encode('utf-8'))
        ratio = bytes_count / len(tokens) if tokens else 0
        print(f"Compression: {bytes_count} bytes → {len(tokens)} tokens (ratio: {ratio:.1f})")

    # Performance test
    print("\n" + "-"*60)
    print("Performance Test")
    print("-"*60)

    long_text = "The quick brown fox jumps over the lazy dog. " * 100

    start_time = time.time()
    tokens = tokenizer.encode(long_text)
    encode_time = time.time() - start_time

    start_time = time.time()
    decoded = tokenizer.decode(tokens)
    decode_time = time.time() - start_time

    print(f"Text length: {len(long_text)} characters")
    print(f"Token count: {len(tokens)} tokens")
    print(f"Encode time: {encode_time*1000:.1f} ms")
    print(f"Decode time: {decode_time*1000:.1f} ms")
    print(f"Encode speed: {len(long_text)/encode_time/1000:.1f} kchar/s")
    print(f"Decode speed: {len(tokens)/decode_time/1000:.1f} ktoken/s")
    print(f"Match: {'✓' if long_text == decoded else '✗'}")

    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
