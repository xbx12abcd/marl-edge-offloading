"""
Setup script for MARL Edge Offloading
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="marl-edge-offloading",
    version="0.1.0",
    author="MARL Edge Offloading Team",
    author_email="your.email@example.com",
    description="Multi-Agent Reinforcement Learning for Edge Computing Task Offloading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/marl-edge-offloading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "gui": [
            "pygame>=2.1.0",
            "PyQt5>=5.15.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.md"],
    },
    entry_points={
        "console_scripts": [
            "marl-edge-train=train_ippo_lowmem:main",
            "marl-edge-eval=evaluate_checkpoint:main",
            "marl-edge-test=test_env:main",
        ],
    },
)