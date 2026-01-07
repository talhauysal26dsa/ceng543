"""
Setup script for Multi-Agent RAG System.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ma-rag-system",
    version="1.0.0",
    author="Multi-Agent RAG Team",
    author_email="team@ma-rag.com",
    description="A comprehensive Retrieval-Augmented Generation system using multiple specialized agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ma-rag-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.7.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "gpu": [
            "torch-gpu>=1.9.0",
            "faiss-gpu>=1.7.0",
        ],
        "web": [
            "streamlit>=1.0.0",
            "gradio>=3.0.0",
        ],
        "monitoring": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ma-rag-preprocess=scripts.preprocess_data:main",
            "ma-rag-benchmark=scripts.run_benchmark:main",
            "ma-rag-analyze=scripts.analyze_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
)
