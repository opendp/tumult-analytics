# pylint: skip-file

import json
import logging
import os
import re
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(".."))


# Project information

project = "Tumult Analytics"
author = "Tumult Labs"
copyright = "Tumult Labs 2022"

package_tag_prefix = "tmlt.analytics"
# A list of version numbers whose docs should not be built. These are used as
# part of a regex, so characters with special meaning in that context (e.g. '.')
# should be escaped, and regex features can be used to suppress multiple
# versions at once. They are not, and cannot easily be, anchored at the end, so
# "1\.2\.3" will suppress version 1.2.3 and all of its pre-releases.
suppressed_versions = [r"1\.2\.0", r"1\.3\.0", r"1\.3\.1"]

# TODO(#1256): Fix import failure in nested class; `tmlt.common`, `tmlt.core` and remove suppress_warnings setting
suppress_warnings = ["autoapi.python_import_resolution", "autodoc.import_object"]
# TODO(#1256): Fix NotebookError caused due to ModuleNotFoundError
nbsphinx_allow_errors = True


# Build information

# Base directory of the build; in CI, this is taken from environment variables;
# locally, it's taken relative to this config file.
project_dir = os.getenv("CI_PROJECT_DIR") or "../.."
# Tag being built for, if any, when running in CI
ci_tag = os.getenv("CI_COMMIT_TAG")
# Name of the version currently being built, e.g. dev, head, or
# tmlt.core-1.4.0. Not set on the initial evaluation of this file with
# sphinx-multiversion, or when using sphinx-build instead.
package_version = os.getenv("SPHINX_MULTIVERSION_NAME")
# Path to the temporary source directory for the version currently being built.
release_sourcedir = os.getenv("SPHINX_MULTIVERSION_SOURCEDIR")
# In linkcheck mode, prepend the intersphinx URLs with the value of BASE_URL_OVERRIDE.
linkcheck_mode_url_prefix = os.getenv("BASE_URL_OVERRIDE")


# Sphinx configuration

extensions = [
    "autoapi.extension",
    "scanpydoc.elegant_typehints",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_multiversion",
    "sphinx_tabs.tabs",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]


# sphinx-multiversion configuration

# Some building blocks for other regexes. These do not actually match all valid
# semantic versions, only the subset that we use.
semver_base_regex = (
    r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
)
semver_prerelease_regex = (
    r"(?:-(?P<prerelease_type>alpha|beta|rc)\.(?P<prerelease_num>0|[1-9]\d*))?"
)
suppressed_versions_regex = (
    rf"(?!{'|'.join(suppressed_versions)})" if suppressed_versions else ""
)

# Build for matching branches from remotes as well as for local branches.
smv_remote_whitelist = r"^.*$"
smv_released_pattern = rf"^refs/tags/{package_tag_prefix}-{semver_base_regex}$"

if ci_tag:
    # Note that this regex matches tags for all packages, not just the current
    # one. This is important because building the docs for a pre-release of one
    # package could rely on the docs for a pre-release of a different package.
    version_regex = re.compile(
        rf"(?P<package>.+-){semver_base_regex}{semver_prerelease_regex}"
    )
    match = version_regex.fullmatch(ci_tag)
    if match is None:
        _logger.error(
            f"Tag {ci_tag} does not match version regex, it appears to be invalid."
        )
        sys.exit(1)

    if match.group("prerelease_type") == "alpha":
        smv_tag_whitelist = (
            rf"{package_tag_prefix}-{suppressed_versions_regex}"
            rf"{semver_base_regex}(-(alpha|beta|rc)\.(0|[1-9]\d*))?$"
        )
    else:
        smv_tag_whitelist = (
            rf"{package_tag_prefix}-{suppressed_versions_regex}"
            rf"{semver_base_regex}(-(beta|rc)\.(0|[1-9]\d*))?$"
        )
    smv_branch_whitelist = "^$"
else:
    smv_branch_whitelist = r"^(dev|main|head)$"
    smv_tag_whitelist = (
        rf"{package_tag_prefix}-{suppressed_versions_regex}"
        rf"{semver_base_regex}(-(beta|rc)\.(0|[1-9]\d*))?$"
    )


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
    jinja_env.globals["package_name"] = package_tag_prefix


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
    ("py:class", "DF"),
    ("py:class", "Row"),
    ("py:class", "tmlt.core.domains.spark_domains.SparkColumnsDescriptor"),
]

# Remove this after intersphinx can use core
nitpick_ignore_regex = [
    (r"py:.*", r"tmlt.core.*"),
]

# Theme settings
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {"collapse_navigation": True, "navigation_depth": 4}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/logo.png"
html_show_sourcelink = False
html_sidebars = {
    "**": [
        "package-name",
        "search-field",
        "sidebar-nav-bs",
        "sidebar-ethical-ads",
        "versions",
    ]
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
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_members)
