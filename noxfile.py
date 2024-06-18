"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.
"""

import os
from pathlib import Path

from tmlt.nox_utils import SessionBuilder

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
        ("3.7-oldest",     "3.7",  "==3.1.1", "==1.8", "==1.2.0", "==0.14.2"),
        ("3.7-pyspark3.2", "3.7",  "==3.2.0", "==1.9", "==1.3.5", ">=0.14.2"),
        ("3.7-newest",     "3.7",  "==3.3.3", "==1.9", "==1.3.5", ">=0.14.2"),
        ("3.8-oldest",     "3.8",  "==3.1.1", "==1.8", "==1.2.0", "==0.14.2"),
        ("3.8-newest",     "3.8",  "==3.5.0", "==1.9", "==1.5.3", ">=0.14.2"),
        ("3.9-oldest",     "3.9",  "==3.1.1", "==1.8", "==1.2.0", "==0.14.2"),
        ("3.9-newest",     "3.9",  "==3.5.0", "==1.9", "==1.5.3", ">=0.14.2"),
        ("3.10-oldest",    "3.10", "==3.1.1", "==1.8", "==1.4.0", "==0.14.2"),
        ("3.10-newest",    "3.10", "==3.5.0", "==1.9", "==1.5.3", ">=0.14.2"),
        ("3.11-oldest",    "3.11", "==3.4.0", "==1.8", "==1.5.0", "==0.14.2"),
        ("3.11-newest",    "3.11", "==3.5.0", "==1.9", "==1.5.3", ">=0.14.2"),
        # fmt: on
    ]
}

LICENSE_IGNORE_GLOBS = [
    r".*\.ci.*",
    r".*\.gitlab.*",
    r".*\.ico",
    r".*\.ipynb",
    r".*\.json",
    r".*\.png",
    r".*\.svg",
]

LICENSE_IGNORE_FILES = [
    r".gitignore",
    r".gitlab-ci.yml",
    r"CONTRIBUTING.md",
    r"LICENSE",
    r"LICENSE.docs",
    r"Makefile",
    r"NOTICE",
    r"noxfile.py",
    r"poetry.lock",
    r"py.typed",
    r"pyproject.toml",
]

LICENSE_KEYWORDS = ["CC-BY-SA-4.0"]
LICENSE_KEYWORDS += ["Apache-2.0"]

ILLEGAL_WORDS_IGNORE_GLOBS = LICENSE_IGNORE_GLOBS
ILLEGAL_WORDS_IGNORE_FILES = LICENSE_IGNORE_FILES
ILLEGAL_WORDS = ["multirepo", "multi-repo"]

AUDIT_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
AUDIT_SUPPRESSIONS = [
    "PYSEC-2023-228",
    # Affects: pip<23.3
    # Notice: Command Injection in pip when used with Mercurial
    # Link: https://github.com/advisories/GHSA-mq26-g339-26xf
    # Impact: None, we don't use Mercurial, and in any case we assume that users will
    #         have their own pip installations -- it is not a dependency of Analytics.
]

# Dictionary mapping benchmark names to the corresponding timeouts
BENCHMARK_TO_TIMEOUT = {
    "keyset_joins": 4,
}


def install_overrides(session):
    """Custom logic run after installing the current package."""
    # Install Core from dist/, if it exists there
    if os.environ.get("CORE_WHEEL_DIR"):
        core_path = Path(os.environ["CORE_WHEEL_DIR"]).resolve()
        core_wheels = list(core_path.glob("*tmlt_core*-cp37*"))
        if len(core_wheels) == 0:
            raise AssertionError(
                "Expected a core wheel since CORE_WHEEL_DIR was set "
                f"(to {os.environ.get('CORE_WHEEL_DIR')}), but didn't find any. "
                f"Instead, found these files in {str(core_path)}: "
                "\n".join([str(path) for path in core_path.glob("*")])
            )
        # Poetry is going to expect, and require, Core version X.Y.Z (ex. "0.6.2"),
        # but the Gitlab-built Core will have a version number
        # X.Y.Z-<some other stuff>-<git commit hash>
        # (ex. "0.6.2-post11+ea346f3")
        # This overrides Poetry's dependencies with our own
        session.poetry.session.install(str(core_wheels[0]))


_builder = SessionBuilder(
    PACKAGE_NAME,
    Path(PACKAGE_SOURCE_DIR).resolve(),
    options={
        "code_dirs": [Path(PACKAGE_SOURCE_DIR).resolve(), Path("test").resolve()],
        "install_overrides": install_overrides,
        "smoketest_script": SMOKETEST_SCRIPT,
        "dependency_matrix": DEPENDENCY_MATRIX,
        "license_exclude_globs": LICENSE_IGNORE_GLOBS,
        "license_exclude_files": LICENSE_IGNORE_FILES,
        "license_keyword_patterns": LICENSE_KEYWORDS,
        "illegal_words_exclude_globs": ILLEGAL_WORDS_IGNORE_GLOBS,
        "illegal_words_exclude_files": ILLEGAL_WORDS_IGNORE_FILES,
        "illegal_words": ILLEGAL_WORDS,
        "audit_versions": AUDIT_VERSIONS,
        "audit_suppressions": AUDIT_SUPPRESSIONS,
        "minimum_coverage": MIN_COVERAGE,
        "coverage_module": "tmlt.analytics",
        "parallel_tests": True,
        "benchmark_to_timeout": BENCHMARK_TO_TIMEOUT,
    },
)

_builder.build()

_builder.black()
_builder.isort()
_builder.mypy()
_builder.pylint()
_builder.pydocstyle()
_builder.license_check()
_builder.illegal_words_check()
_builder.audit()

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

_builder.benchmark()
_builder.benchmark_dependency_matrix()

_builder.release_test()
_builder.release_smoketest()

_builder.prepare_release()
_builder.post_release()
