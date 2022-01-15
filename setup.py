#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fnmatch
from setuptools import setup, find_packages


# Path to the directory that contains this setup.py file.
base_dir = os.path.abspath(os.path.dirname(__file__))

def find_files(directory):
    matches = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

    return matches


setup(
    name="meeko",
    version='0.3.0',
    author="Forli Lab",
    author_email="forli@scripps.edu",
    url="https://github.com/ccsb-scripps/meeko",
    description='Python package for preparing small molecule for docking',
    long_description=open(os.path.join(base_dir, 'README.md')).read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    scripts=["scripts/mk_prepare_ligand.py",
             "scripts/mk_copy_coords.py"],
           #"scripts/dry.py",
           #"scripts/mapwater.py",
           #"scripts/wet.py"],
    package_data={"meeko" : ["data/*"]},
    data_files=[("", ["README.md", "LICENSE"]),
                ("scripts", find_files("scripts"))],
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy>=1.18'],
    python_requires='>=3.5.*',
    license="Apache-2.0",
    keywords=["molecular modeling", "drug design",
            "docking", "autodock"],
    classifiers=[
        'Environment :: Console',
        'Environment :: Other Environment',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        #'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries'
    ]
)
