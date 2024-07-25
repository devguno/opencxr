#!/usr/bin/env python

from setuptools import find_packages, setup

NAME = "opencxr"
REQUIRES_PYTHON = ">=3.7"

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "wget",
    "tensorflow>=2.0.0",  # Ensure a recent version of TensorFlow
    "SimpleITK",
    "pydicom",
    "pypng",
    "scikit-image",
    "scikit-build",
    "numpy",
]

setup(
    name=NAME,
    author="Keelin Murphy",
    author_email="keelin.murphy@radboudumc.nl",
    description="a collection of algorithms for processing of chest radiograph (CXR) images",
    install_requires=requirements,
    license="Apache 2.0",
    long_description=readme,
    keywords="opencxr",
    url="https://github.com/DIAGNijmegen/opencxr",
    packages=find_packages(),
    version="1.2.0",
)
