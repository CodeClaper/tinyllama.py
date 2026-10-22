from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

long_description = (ROOT / "README.md").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="tinyllama.py",
    version="0.1.0",
    description="A lightweight and fast inference engine based on vLLM with an OpenAI-compatible API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CodeClaper",
    url="https://github.com/CodeClaper/tinyllama.py",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.12",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["llm", "inference", "vllm", "openai", "fastapi"],
    zip_safe=False,
)
