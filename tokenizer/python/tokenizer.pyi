"""Type stubs for tokenizer module"""

from typing import List

class Tokenizer:
    """Byte Pair Encoding (BPE) Tokenizer"""

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size"""
        ...

    @property
    def num_merges(self) -> int:
        """Get the number of merges"""
        ...

    @property
    def num_special_tokens(self) -> int:
        """Get the number of special tokens"""
        ...

    def __init__(self, pattern: str | None = None) -> None:
        """
        Create a new tokenizer with optional regex pattern

        Args:
            pattern: Optional regex pattern for tokenization. If None, uses default GPT-4 pattern
        """
        ...

    def load_from_huggingface(self, source: str) -> None:
        """
        Load vocabulary and merges from a HuggingFace tokenizer.json file or URL

        Args:
            source: URL (http:// or https://) or local file path to tokenizer.json

        Raises:
            IOError: If file cannot be read or downloaded
            ValueError: If tokenizer format is invalid
            NotImplementedError: If tokenizer type is not supported (only BPE is supported)
        """
        ...

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a sequence of token IDs

        Args:
            text: The text to encode

        Returns:
            List of token IDs
        """
        ...

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back into text

        Args:
            ids: List of token IDs to decode

        Returns:
            The decoded text string
        """
        ...
