# pylint: skip-file

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

### Project information

project = "Tumult Analytics"
author = "Tumult Labs"
copyright = "2025"
# Note that this is the name of the module provided by the package, not
# necessarily the name of the package as pip understands it.
package_name = "tmlt"

default_branch = "main"

### Build information

pipeline_type = os.getenv("GITHUB_REF_TYPE")
ref_name = os.getenv("GITHUB_REF_NAME")

# For non-prerelease tags, make the version "vX.Y" to match how we show it in
# the version switcher and the docs URLs. Sphinx's nomenclature around versions
# can be a bit confusing -- "version" means sort of the documentation version
# (for us, the minor release), while "release" is the full version number of the
# package on which the docs were built.
if pipeline_type and pipeline_type == "tag" and "-" not in ref_name:
    release = ref_name
    version = "v" + ".".join(ref_name.split(".")[:2])
elif ref_name and ref_name == default_branch:
    release = version = "dev"
else:
    release = version = ref_name or "HEAD"

commit_sha = os.getenv("GITHUB_SHA")

commit_hash = commit_sha[:8] if commit_sha else "unknown version"
build_time = datetime.datetime.utcnow().isoformat(sep=" ", timespec="minutes")

# Linkcheck will complain that these anchors don't exist,
# even though the link works.
linkcheck_ignore = [
    "https://colab.research.google.com/drive/1hIbp7y1uXIXc-MeiCAV4_0EwgSZzoM8U#offline=true&sandboxMode=true",
    "https://docs.databricks.com/release-notes/runtime/releases.html",
    "http://creativecommons.org/licenses/by-sa/4.0/",
]

### Sphinx configuration

extensions = [
    "sphinxcontrib.images",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # smart_resolver fixes cases where an object is documented under a name
    # different from its qualname, e.g. due to importing it in an __init__.
    "sphinx_automodapi.smart_resolver",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]

bibtex_bibfiles = []

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

# Autosummary settings
autosummary_generate = True

# Autodoc settings
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}

# General settings
master_doc = "index"
exclude_patterns = ["templates"]
# Don't test stand-alone doctest blocks -- this prevents the examples from
# docstrings from being tested by Sphinx (nosetests --with-doctest already
# covers them).
doctest_test_doctest_blocks = ""

nitpick_ignore = [
    # TODO(#3216): These private base classes are going away, ignore them for now.
    ("py:obj", "tmlt.analytics.metrics._base.ScalarMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.MeasureColumnMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.SingleBaselineMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.MultiBaselineMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.GroupedMetric"),
    # TypeVar support: https://github.com/tox-dev/sphinx-autodoc-typehints/issues/39
    ("py:obj", "tmlt.analytics.binning_spec.BinT"),
    ("py:obj", "tmlt.analytics.binning_spec.BinNameT"),
    # TODO remove
    ("py:class", "pandas.core.frame.DataFrame"),
]

# Remove this after intersphinx can use core
nitpick_ignore_regex = [(r"py:.*", r"tmlt.core.*")]

json_url = "https://opendp.github.io/tumult-docs/analytics/versions.json"

# Theme settings
templates_path = ["_templates", "_templates/autosummary"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "back_to_top_button": False,
    "collapse_navigation": True,
    "footer_start": ["copyright", "build-info"],
    "footer_end": ["sphinx-version", "theme-version"],
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/opendp/tumult-analytics",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Slack",
            "url": "https://join.slack.com/t/opendp/shared_invite/zt-1aca9bm7k-hG7olKz6CiGm8htI2lxE8w",
            "icon": "fab fa-slack",
            "type": "fontawesome",
        },
    ],
    "navigation_depth": 4,
    "navbar_end": ["navbar-icon-links"],
    "switcher": {
        "json_url": json_url,
        "version_match": version,
    },
    "show_nav_level": 2,
    "show_toc_level": 3,
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
html_sidebars = {"**": ["package-name", "version-switcher", "left-sidebar"]}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/1.18/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/1.2.0/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "pyspark": ("https://archive.apache.org/dist/spark/docs/3.1.1/api/python/", None),
}

# Substitutions
rst_epilog = f"""
.. |project| replace:: {project}
"""

# Customizing what gets shown in API docs

# Members shown for classes whose documentation is split across multiple pages.
showed_members = {
    "Session": [],
    "QueryBuilder": ["__init__", "clone"],
    "GroupedQueryBuilder": ["__init__"],
}
# Classes for which we want to show the __init__ method.
show_init = {"BinningSpec"}

# Methods that are show directly after certain methods
aggs = [
    "average",
    "count",
    "count_distinct",
    "get_bounds",
    "max",
    "median",
    "min",
    "quantile",
    "stdev",
    "sum",
    "variance",
]
companion_methods = {
    f"QueryBuilder.{agg}": f"GroupedQueryBuilder.{agg}" for agg in aggs
}
# Classes that are shown directly after certain methods
companion_classes = {
    # Aggregation mechanisms
    "QueryBuilder.average": "AverageMechanism",
    "QueryBuilder.count": "CountMechanism",
    "QueryBuilder.count_distinct": "CountDistinctMechanism",
    "QueryBuilder.stdev": "StdevMechanism",
    "QueryBuilder.sum": "SumMechanism",
    "QueryBuilder.variance": "VarianceMechanism",
    # Private join truncation
    "QueryBuilder.join_private": "TruncationStrategy",
}

autosummary_context = {
    "show_init": show_init,
    "showed_members": showed_members,
    "companion_methods": companion_methods,
    "companion_classes": companion_classes,
}


def setup(sphinx):
    # Write out the version and release numbers (using Sphinx's definitions of
    # them) for use by later automation.
    outdir = Path(sphinx.outdir)
    (outdir / "_version").write_text(version)
    (outdir / "_release").write_text(release)
