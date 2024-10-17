# pylint: skip-file

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import datetime
import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

### Project information

project = "Tumult Analytics"
author = "Tumult Labs"
copyright = "2024 Tumult Labs"
# Note that this is the name of the module provided by the package, not
# necessarily the name of the package as pip understands it.
package_name = "tmlt.analytics"

# TODO(#1256): Fix import failure in nested class; `tmlt.core` and remove
#     suppress_warnings setting
suppress_warnings = ["autoapi.python_import_resolution", "autodoc.import_object"]


### Build information

ci_tag = os.getenv("CI_COMMIT_TAG")
ci_branch = os.getenv("CI_COMMIT_BRANCH")

# For non-prerelease tags, make the version "vX.Y" to match how we show it in
# the version switcher and the docs URLs. Sphinx's nomenclature around versions
# can be a bit confusing -- "version" means sort of the documentation version
# (for us, the minor release), while "release" is the full version number of the
# package on which the docs were built.
if ci_tag and "-" not in ci_tag:
    release = ci_tag
    version = "v" + ".".join(ci_tag.split(".")[:2])
else:
    release = version = ci_tag or ci_branch or "HEAD"

commit_hash = os.getenv("CI_COMMIT_SHORT_SHA") or "unknown version"
build_time = datetime.datetime.utcnow().isoformat(sep=" ", timespec="minutes")

# Linkcheck will complain that these anchors don't exist,
# even though the link works.
linkcheck_ignore = [
    "https://colab.research.google.com/drive/18J_UrHAKJf52RMRxi4OOpk59dV9tvKxO#offline=true&sandboxMode=true",
    "https://docs.databricks.com/release-notes/runtime/releases.html",
]

### Sphinx configuration

extensions = [
    "autoapi.extension",
    "sphinxcontrib.images",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
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

# Autoapi settings
autoapi_root = "reference"
autoapi_dirs = ["../tmlt/"]
# TODO(#3320): Don't include Tune shortcuts in the documentation for now.
autoapi_ignore = ["*tmlt/tune/*"]
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
    jinja_env.tests["is_mixin_class"] = lambda classname: classname.endswith("Mixin")
    jinja_env.tests["is_base_builder"] = (
        lambda classname: classname == "tmlt.analytics._base_builder.BaseBuilder"
    )


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

nitpick_ignore = [
    # TODO(#3216): These private base classes are going away, ignore them for now.
    ("py:obj", "tmlt.analytics.metrics._base.ScalarMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.MeasureColumnMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.SingleBaselineMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.MultiBaselineMetric"),
    ("py:obj", "tmlt.analytics.metrics._base.GroupedMetric"),
    # TODO(#1629): Schema is currently private, even though it appears in the
    #     public docs in a couple of places. This ignore can be removed once it
    #     is public.
    ("py:class", "tmlt.analytics._schema.Schema"),
    # TODO(#1357): We shouldn't be showing the types for these values at all,
    #     but they're appearing at the end of class docstrings because
    #     __init__'s parameters are documented there for some reason. Once we
    #     fix that, it should be possible to remove this ignore.
    ("py:class", "tmlt.analytics._table_identifier.Identifier"),
    # TypeVar support: https://github.com/tox-dev/sphinx-autodoc-typehints/issues/39
    ("py:class", "Row"),
    ("py:class", "BinT"),
    ("py:class", "tmlt.analytics.binning_spec.BinT"),
    ("py:class", "BinNameT"),
    # Ellipses (Tuple[blah, ...]) confuse Sphinx apparently; suppressing warning
    ("py:class", "Ellipsis"),
]

# Remove this after intersphinx can use core
nitpick_ignore_regex = [(r"py:.*", r"tmlt.core.*")]

json_url = "https://docs.tmlt.dev/analytics/versions.json"

# Theme settings
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "header_links_before_dropdown": 6,
    "collapse_navigation": True,
    "navigation_depth": 4,
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright", "build-info"],
    "footer_end": ["sphinx-version", "theme-version"],
    "switcher": {
        "json_url": json_url,
        "version_match": version,
    },
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://gitlab.com/tumult-labs/analytics",
            "icon": "fab fa-gitlab",
            "type": "fontawesome",
        },
        {
            "name": "Slack",
            "url": "https://tmlt.dev/slack",
            "icon": "fab fa-slack",
            "type": "fontawesome",
        },
    ],
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
html_sidebars = {"**": ["package-name", "version-switcher", "sidebar-nav-bs"]}

# Intersphinx mapping

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/1.18/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/1.2.0/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "pyspark": ("https://archive.apache.org/dist/spark/docs/3.1.1/api/python/", None),
}


# Substitutions

rst_epilog = """
.. |PRO| raw:: html

    <a href="https://tmlt.dev" style="text-decoration : none">
        <img src="https://img.shields.io/badge/PRO-c53a58" alt="This is only applicable to Analytics Pro." title="This is only available in Analytics Pro">
    </a>
.. |PRO_NOTE| replace:: This is only available on a paid version of Tumult Analytics. If you
    would like to hear more, please contact us at info@tmlt.io.

.. |project| replace:: {}
""".format(
    project
)


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
        "__init_subclass__",
    ]
    excluded_attributes = ["__slots__"]
    if what == "method" and name.split(".")[-1] in excluded_methods:
        return True
    if what == "attribute" and name.split(".")[-1] in excluded_attributes:
        return True
    if "@nodoc" in obj.docstring:
        return True
    # TODO(#3320): Don't include top-level shortcuts in the docs for now.
    if what not in ["module", "package"] and len(name.split(".")) == 3:
        # Only top-level object currently in the docs, keep it.
        if name == "tmlt.analytics.AnalyticsInternalError":
            return skip
        return True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_members)
    # Write out the version and release numbers (using Sphinx's definitions of
    # them) for use by later automation.
    outdir = Path(sphinx.outdir)
    (outdir / "_version").write_text(version)
    (outdir / "_release").write_text(release)
