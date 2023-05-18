# pylint: skip-file

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import logging
import os
import sys
import datetime

_logger = logging.getLogger(__name__)

# Project information

project = "Tumult Analytics"
author = "Tumult Labs"
copyright = "Tumult Labs 2023"

package_name = "tmlt.analytics"

# TODO(#1256): Fix import failure in nested class; `tmlt.core` and remove suppress_warnings setting
suppress_warnings = ["autoapi.python_import_resolution", "autodoc.import_object"]


# Build information

ci_tag = os.getenv("CI_COMMIT_TAG")
ci_branch = os.getenv("CI_COMMIT_BRANCH")

version = ci_tag or ci_branch or "HEAD"
commit_hash = os.getenv("CI_COMMIT_SHORT_SHA") or "unknown version"
build_time = datetime.datetime.utcnow().isoformat(sep=" ", timespec="minutes")

# Linkcheck will complain that these anchors don't exist,
# even though the link works.
linkcheck_ignore = [
    "https://colab.research.google.com/drive/18J_UrHAKJf52RMRxi4OOpk59dV9tvKxO#offline=true&sandboxMode=true"
]

# Sphinx configuration

extensions = [
    "autoapi.extension",
    "scanpydoc.elegant_typehints",
    "sphinxcontrib.images",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_panels",
]

# Prevent sphinx_panels from loading bootstrap a second time
panels_add_bootstrap_css = False
# Change colors & contrast to inactive tab labels so they pass WCAG AA; all
# other colors are the same as the defaults:
#   https://sphinx-panels.readthedocs.io/en/latest/#tabbed-content
panels_css_variables = {
    "tabs-color-label-active": "hsla(231, 99%, 66%, 1)",
    "tabs-color-label-inactive": "rgba(135, 138, 150, 1)",
    "tabs-color-overline": "rgb(207, 236, 238)",
    "tabs-color-underline": "rgb(207, 236, 238)",
    "tabs-size-label": "1rem",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autoapi settings
autoapi_root = "reference"
autoapi_dirs = ["../tmlt/"]
autoapi_keep_files = False
autoapi_template_dir = "../doc/templates"
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = True  # This is important for intersphinx
autoapi_options = [
    "members",
    "show-inheritance",
    "special-members",
    "show-module-summary",
    "imported-members",
    "inherited-members",
]
add_module_names = False


def autoapi_prepare_jinja_env(jinja_env):
    # Set the package_name variable so it can be used in templates.
    jinja_env.globals["package_name"] = package_name
    # Define a new test for filtering out objects with @nodoc in their
    # docstring; this needs to be defined here because Jinja2 doesn't have a
    # built-in "contains" or "match" test.
    jinja_env.tests["nodoc"] = lambda obj: "@nodoc" in obj.docstring


# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# General settings
master_doc = "index"
exclude_patterns = ["templates"]
# Don't test stand-alone doctest blocks -- this prevents the examples from
# docstrings from being tested by Sphinx (nosetests --with-doctest already
# covers them).
doctest_test_doctest_blocks = ""

# scanpydoc overrides to resolve target
qualname_overrides = {
    "sp.Expr": "sympy.core.expr.Expr",
    "pyspark.sql.dataframe.DataFrame": "pyspark.sql.DataFrame",
    "pyspark.sql.session.SparkSession": "pyspark.sql.SparkSession",
}

nitpick_ignore = [
    # Expr in __init__ is resolved fine but not in type hint
    ("py:class", "sympy.Expr"),
    # Optional SparkSession in NoiseFreeQueryEvaluator __init__ is not resolved
    ("py:class", "pyspark.sql.session.SparkSession"),
    # Sphinx can't resolve DataFrame in KeySet.__init__
    ("py:class", "pyspark.sql.dataframe.DataFrame"),
    # TypeVar support: https://github.com/agronholm/sphinx-autodoc-typehints/issues/39
    ("py:class", "Ellipsis"),
    ("py:class", "DF"),
    ("py:class", "Row"),
    ("py:class", "BinT"),
    ("py:class", "tmlt.analytics.binning_spec.BinT"),
    ("py:class", "BinNameT"),
    ("py:class", "Generic[BinT, BinNameT]"),
    ("py:class", "tmlt.core.domains.spark_domains.SparkColumnsDescriptor"),
    ("py:class", "sympy.core.expr.Expr"),
    ("py:class", "Epsilon"),
    ("py:class", "Delta"),
    ("py:class", "Rho"),
]

# Remove this after intersphinx can use core
nitpick_ignore_regex = [(r"py:.*", r"tmlt.core.*")]

# Theme settings
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["copyright", "build-info", "sphinx-version"],
    "switcher": {
        "json_url": "https://docs.tmlt.dev/analytics/versions.json",
        "version_match": version,
    },
    "gitlab_url": "https://gitlab.com/tumult-labs/analytics",
}
html_context = {
    "default_mode": "light",
    "commit_hash": commit_hash,
    "build_time": build_time,
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/version-banner.js"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False
html_sidebars = {
    "**": ["package-name", "version-switcher", "search-field", "sidebar-nav-bs"]
}

# Intersphinx mapping

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/1.18/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/1.2.0/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "pyspark": ("https://spark.apache.org/docs/3.0.0/api/python/", None),
}


def skip_members(app, what, name, obj, skip, options):
    """Skip some members."""
    excluded_methods = [
        "__dir__",
        "__format__",
        "__hash__",
        "__post_init__",
        "__reduce__",
        "__reduce_ex__",
        "__repr__",
        "__setattr__",
        "__sizeof__",
        "__str__",
        "__subclasshook__",
    ]
    excluded_attributes = ["__slots__"]
    if what == "method" and name.split(".")[-1] in excluded_methods:
        return True
    if what == "attribute" and name.split(".")[-1] in excluded_attributes:
        return True
    if "@nodoc" in obj.docstring:
        return True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_members)
