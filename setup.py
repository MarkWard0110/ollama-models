from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open(os.path.join("ollama_models", "__init__.py"), "r") as f:
    content = f.read()
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", content)
    version = version_match.group(1) if version_match else "0.0.0"

# Read long description from README
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ollama-models",
    version=version,
    author="Binary Ward",
    description="Tools for managing and analyzing Ollama models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/binaryward/ollama-models",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ollama-models=ollama_models.cli:main",
        ],
    },
)
