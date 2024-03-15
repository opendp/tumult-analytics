"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.
"""

import os
from glob import glob
from pathlib import Path
import sys

from nox_poetry import session as poetry_session

PACKAGE_NAME = "tmlt.analytics"
"""Name of the package."""
PACKAGE_SOURCE_DIR = "tmlt/analytics"
"""Relative path from the project root to its source code."""
SMOKETEST_SCRIPT = """
from tmlt.analytics.utils import check_installation
check_installation()
"""
"""Python script to run as a quick self-test."""

MIN_COVERAGE = 75
"""For test suites where we track coverage (i.e. the fast tests and the full
test suite), fail if test coverage falls below this percentage."""

DEPENDENCY_MATRIX = {
    name: {
        # The Python minor version to run with
        "python": python,
        # All other entries take PEP440 version specifiers for the package named in
        # the key -- see https://peps.python.org/pep-0440/#version-specifiers
        "pyspark[sql]": pyspark,
        "sympy": sympy,
        "pandas": pandas,
        "tmlt.core": core,
    }
    for (name, python, pyspark, sympy, pandas, core) in [
        # fmt: off
        ("3.7-oldest",     "3.7",  "==3.0.0", "==1.8", "==1.2.0", "==0.12.0"),
        ("3.7-pyspark3.1", "3.7",  "==3.1.1", "==1.9", "==1.3.5", ">=0.12.0"),
        ("3.7-pyspark3.2", "3.7",  "==3.2.0", "==1.9", "==1.3.5", ">=0.12.0"),
        ("3.7-newest",     "3.7",  "==3.3.3", "==1.9", "==1.3.5", ">=0.12.0"),
        ("3.8-oldest",     "3.8",  "==3.0.0", "==1.8", "==1.2.0", "==0.12.0"),
        ("3.8-newest",     "3.8",  "==3.5.0", "==1.9", "==1.5.3", ">=0.12.0"),
        ("3.9-oldest",     "3.9",  "==3.0.0", "==1.8", "==1.2.0", "==0.12.0"),
        ("3.9-newest",     "3.9",  "==3.5.0", "==1.9", "==1.5.3", ">=0.12.0"),
        ("3.10-oldest",    "3.10", "==3.0.0", "==1.8", "==1.4.0", "==0.12.0"),
        ("3.10-newest",    "3.10", "==3.5.0", "==1.9", "==1.5.3", ">=0.12.0"),
        ("3.11-oldest",    "3.11", "==3.4.0", "==1.8", "==1.5.0", "==0.12.0"),
        ("3.11-newest",    "3.11", "==3.5.0", "==1.9", "==1.5.3", ">=0.12.0"),
        # fmt: on
    ]
}


def install_overrides(session):
    """Custom logic run after installing the current package."""
    # Install Core from dist/, if it exists there
    if os.environ.get("PARENT_PIPELINE_ID"):
        core_wheels = glob(r"./dist/*tmlt_core*-cp37*")
        if len(core_wheels) == 0:
            raise AssertionError(
                "Expected a core wheel since PARENT_PIPELINE_ID was set "
                f"(to {os.environ.get('PARENT_PIPELINE_ID')}), but didn't find any. "
                f"There should be one in dist/, which contains: {glob(r'dist/*')}, "
            )
        # Poetry is going to expect, and require, Core version X.Y.Z (ex. "0.6.2"),
        # but the Gitlab-built Core will have a version number
        # X.Y.Z-<some other stuff>-<git commit hash>
        # (ex. "0.6.2-post11+ea346f3")
        # This overrides Poetry's dependencies with our own
        session.poetry.session.install(core_wheels[0])


CWD = Path(".").resolve()
sys.path.append(str(CWD / ".nox-utils"))
from nox_utils import SessionBuilder

_builder = SessionBuilder(
    PACKAGE_NAME,
    Path(PACKAGE_SOURCE_DIR).resolve(),
    options={
        "code_dirs": [Path(PACKAGE_SOURCE_DIR).resolve(), Path("test").resolve()],
        "install_overrides": install_overrides,
        "smoketest_script": SMOKETEST_SCRIPT,
        "dependency_matrix": DEPENDENCY_MATRIX,
        "minimum_coverage": MIN_COVERAGE,
        "coverage_module": "tmlt.analytics",
    },
)

_builder.build()

_builder.black()
_builder.isort()
_builder.mypy()
_builder.pylint()
_builder.pydocstyle()

_builder.test()
_builder.test_doctest()
_builder.test_demos()
_builder.test_smoketest()
_builder.test_fast()
_builder.test_slow()
_builder.test_dependency_matrix()

_builder.docs_linkcheck()
_builder.docs_doctest()
_builder.docs()

_builder.release_test()
_builder.release_smoketest()

_builder.prepare_release()
_builder.post_release()

