# read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pyflox",
    packages=find_packages(include=["flox"]),
    version="0.1.5",
    description="Library for serverless Federated Learning experiments.",
    readme="README.md",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikita-kotsehub/FLoX",
    download_url="https://github.com/nikita-kotsehub/FLoX/archive/refs/tags/v0.1.3-test.tar.gz",
    author="Nikita Kotsehub",
    author_email="mykyta.kotsehub@gmail.com",
    license="MIT",
    install_requires=["numpy", "funcx", "parsl"],
    keywords=["federated_learning", "serverless", "edge_devices"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.0",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
