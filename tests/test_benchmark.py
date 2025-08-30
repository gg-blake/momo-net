import time
import tiktoken
from tokenizer import Tokenizer
import pytest

url = "https://huggingface.co/BEE-spoke-data/cl100k_base-mlm/resolve/main/tokenizer.json"
enc = Tokenizer()
enc.load_from_huggingface(url)

@pytest.fixture(scope="session")
def texts():
    short_text = "Hello world! This is a short sentence."
    medium_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
    long_text = " ".join(["Lorem ipsum dolor sit amet, consectetur adipiscing elit."] * 2000)
    return {
        "short": short_text,
        "medium": medium_text,
        "long": long_text,
    }


@pytest.fixture(scope="session")
def tiktoken_encode():
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode

@pytest.fixture(scope="session")
def custom_encode():
    return enc.encode

@pytest.mark.benchmark(group="encode")
@pytest.mark.parametrize("size", ["short", "medium", "long"])
def test_my_tokenizer_encode(benchmark, texts, custom_encode, size):
    text = texts[size]
    benchmark(custom_encode, text)


@pytest.mark.benchmark(group="encode")
@pytest.mark.parametrize("size", ["short", "medium", "long"])
def test_tiktoken_encode(benchmark, texts, tiktoken_encode, size):
    text = texts[size]
    benchmark(tiktoken_encode, text)
        