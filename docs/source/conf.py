# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


project = 'Tortreinador'
copyright = '2024, Xavier'
author = 'Xavier'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # 自动生成文档的扩展
    'sphinx.ext.napoleon',  # 支持 Google 和 NumPy 风格的 docstring
    'sphinx.ext.viewcode',     # 在文档中添加源码链接
    'sphinx.ext.autosummary',  # 自动生成模块摘要
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# html_static_path = ['_static']
