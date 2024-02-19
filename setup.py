#!/usr/bin/env python3
from glob import glob

from setuptools import find_packages, setup

setup(
    name="cached_opentims",
    version="0.0.1",
    url="https://github.com/michalsta/cached_opentims",
    author="Michał Startek",
    author_email="author@gmail.com",
    description="Memory-mapped, on-disk-extraced, indexed access to TIMS TOF data. Ay comrad.",
    packages=find_packages(),
    install_requires=[
        "mmapped_df",
        "numba",
        "numba_progress",
        "numpy",
        "pandas",
    ],
    scripts=glob("tools/*.py"),
)
