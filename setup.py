#!/usr/bin/env python3
from setuptools import setup, find_packages
from glob import glob


setup(
    name="cached_opentims",
    version="0.0.1",
    url="https://github.com/michalsta/cached_opentims",
    author="Michał Startek",
    author_email="author@gmail.com",
    description="Memory-mapped, on-disk-extraced, indexed access to TIMS TOF data",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "mmapped_df"],
    scripts=glob("tools/*.py"),
)
