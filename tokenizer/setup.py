from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="tokenizer",
    version="0.1.0",
    description="Fast BPE tokenizer implementation in Rust with Python bindings",
    author="Your Name",
    author_email="your.email@example.com",
    rust_extensions=[
        RustExtension(
            "tokenizer.tokenizer",
            binding=Binding.PyO3,
            path="Cargo.toml",
        )
    ],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "setuptools-rust>=1.5.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
