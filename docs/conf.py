"""Sphinx configuration file for SmileyLlama API documentation."""

import os
import sys
from pathlib import Path

# Add the project root to the path so Sphinx can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Project information
project = "SmileyLlama"
copyright = "2024, Joe C. Cavanagh, Eric Wang"
author = "Joe C. Cavanagh, Eric Wang"
release = "0.2.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxcontrib.autodoc_pydantic",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_inherit_docstrings = False
autodoc_member_order = 'bysource'

# Autodoc Pydantic settings
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_field_show_alias = True
autodoc_pydantic_field_show_default = True

# Autosummary settings
autosummary_generate = True

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Templates
templates_path = ["_templates"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rdkit": ("https://www.rdkit.org/docs/", None),
}

