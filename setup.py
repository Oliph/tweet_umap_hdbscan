#!/usr/bin/env python
from distutils.core import setup

setup(
    name="umap_hdbscan",
    version="0.1",
    description="tools to create clusters based on umap for dimension reduction and HDBSCAN for clustering",
    author="Olivier Philippe",
    author_email="olivier.philippe@gmail.com",
    packages=setuptools.find_packages(),
    py_modules=["umap_hdbscan"],
)
