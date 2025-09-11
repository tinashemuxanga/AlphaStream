"""
AlphaStream Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="alphastream",
    version="1.0.0",
    author="AlphaStream Team",
    author_email="contact@alphastream.ai",
    description="ML-powered trading signal generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alphastream",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.19.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alphastream=train_models:main",
            "alphastream-api=api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "alphastream": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
)
