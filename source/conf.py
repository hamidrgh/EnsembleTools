import os
import sys
sys.path.insert(0, os.path.abspath('/home/hamid/Desktop/EnsembleTools')) 


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IDPET'
copyright = '2024, Hamidreza Ghafouri, Giacomo Janson'
author = 'Hamidreza Ghafouri, Giacomo Janson'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Enables autodoc
    'sphinx.ext.napoleon',   # Enables Google-style and NumPy-style docstrings
    'sphinx_rtd_theme',      # Read the Docs theme
    'sphinx.ext.mathjax'
]

templates_path = ['_templates']
exclude_patterns = ['.gitignore', ]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
