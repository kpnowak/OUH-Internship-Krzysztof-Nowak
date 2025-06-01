"""
Multi-Omics Data Fusion Optimization Pipeline
Setup script for package installation with optional dependencies.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Multi-Omics Data Fusion Optimization Pipeline"

# Core dependencies - essential for basic functionality
CORE_REQUIREMENTS = [
    "numpy>=1.21.0",
    "pandas>=1.3.0", 
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "xgboost>=1.6.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "joblib>=1.1.0",
    "threadpoolctl>=3.0.0",
    "psutil>=5.8.0",
    "boruta>=0.3.0",
]

# Note: All other required modules are part of Python's standard library:
# logging, os, sys, time, warnings, hashlib, copy, gc, threading, random,
# re, datetime, pathlib, argparse, subprocess, importlib, functools, 
# contextlib, collections, itertools, typing, difflib, traceback, dataclasses

# Visualization dependencies - enhanced plotting capabilities
VISUALIZATION_REQUIREMENTS = [
    "scikit-posthocs>=0.6.0",  # Critical difference diagrams for MAD analysis
]

# Development dependencies - testing and code quality tools
DEVELOPMENT_REQUIREMENTS = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
]

# Optional extras for different installation types
EXTRAS_REQUIRE = {
    "visualization": VISUALIZATION_REQUIREMENTS,
    "development": DEVELOPMENT_REQUIREMENTS,
    "all": VISUALIZATION_REQUIREMENTS + DEVELOPMENT_REQUIREMENTS,
}

setup(
    name="data-fusion-pipeline",
    version="1.0.0",
    author="Krzysztof Nowak",
    author_email="krzysztof.nowak.krakow@gmail.com",
    description="Multi-Omics Data Fusion Optimization Pipeline for Cancer Detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/data-fusion-pipeline",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "data-fusion=main:main",
            "mad-analysis=mad_analysis:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning, bioinformatics, multi-omics, data-fusion, cancer-detection, feature-selection, dimensionality-reduction",
    project_urls={
        "Bug Reports": "https://github.com/your-username/data-fusion-pipeline/issues",
        "Source": "https://github.com/your-username/data-fusion-pipeline",
        "Documentation": "https://github.com/your-username/data-fusion-pipeline/blob/main/README.md",
    },
) 