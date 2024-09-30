# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../meeko/'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'meeko'
copyright = '2024, Forli Lab at Scripps Research'
author = 'The Meeko authors'
#release = '0.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

html_logo = "images/raccoon.png"


templates_path = ['_templates']
exclude_patterns = []

pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'

html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/forlilab/meeko',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}
