#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import fnmatch
from setuptools import setup, find_packages


def find_files(directory):
    matches = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

    return matches


setup(name="meeko",
      version=0.1,
      description="Meeko",
      author="Stefano Forli",
      author_email="forli@scripps.edu",
      url="https://github.com/ccsb-scripps/meeko",
      packages=find_packages(),
      scripts=["scripts/prepare_macrocycle.py",
               "scripts/dry.py",
               "scripts/mapwater.py",
               "scripts/wet.py"],
      package_data={
            "meeko" : ["data/*"]
      },
      data_files=[("", ["README.md", "LICENSE"]),
                  ("scripts", find_files("scripts"))],
      include_package_data=True,
      zip_safe=False,
      license="Apache-2.0",
      keywords=["molecular modeling", "drug design",
                "docking", "autodock"],
      classifiers=["Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9"
                   "Operating System :: Unix",
                   "Operating System :: MacOS",
                   "Topic :: Scientific/Engineering"]
)
