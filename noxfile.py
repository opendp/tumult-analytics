"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.
"""

import os
from pathlib import Path

import nox
from tmlt.nox_utils import SessionManager

nox.options.default_venv_backend = "uv|virtualenv"

CWD = Path(".").resolve()

PACKAGE_NAME = "tmlt.analytics"
"""Name of the package."""
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
        ("3.9-oldest",     "3.9",  "==3.3.1", "==1.8", "==1.4.0", "==0.18.0"),
        ("3.9-pyspark3.4", "3.9",  "==3.4.0", "==1.9", "==1.5.3", ">=0.18.0"),
        ("3.9-newest",     "3.9",  "==3.5.1", "==1.9", "==1.5.3", ">=0.18.0"),
        ("3.10-oldest",    "3.10", "==3.3.1", "==1.8", "==1.4.0", "==0.18.0"),
        ("3.10-newest",    "3.10", "==3.5.1", "==1.9", "==1.5.3", ">=0.18.0"),
        ("3.11-oldest",    "3.11", "==3.4.0", "==1.8", "==1.5.0", "==0.18.0"),
        ("3.11-newest",    "3.11", "==3.5.1", "==1.9", "==1.5.3", ">=0.18.0"),
        ("3.12-oldest",    "3.12", "==3.5.0", "==1.8", "==2.2.0", "==0.18.0"),
        ("3.12-newest",    "3.12", "==3.5.1", "==1.9", "==2.2.3", ">=0.18.0"),
        # fmt: on
    ]
}

AUDIT_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]
AUDIT_SUPPRESSIONS = [
    "PYSEC-2023-228",
    # Affects: pip<23.3
    # Notice: Command Injection in pip when used with Mercurial
    # Link: https://github.com/advisories/GHSA-mq26-g339-26xf
    # Impact: None, we don't use Mercurial, and in any case we assume that users will
    #         have their own pip installations -- it is not a dependency of Analytics.
    "PYSEC-2017-147",
    # Affects: PySpark 1.6 through 2.1
    # Link: https://nvd.nist.gov/vuln/detail/CVE-2017-12612
    # Impact: None, we don't support these versions of PySpark. This appears to
    #         be showing up due to a bad data import into the PyPA vulnerability
    #         database [0], which they are aware of and working to fix [1], but
    #         in the mean time we are also ignoring it here.
    # [0] https://github.com/pypa/advisory-database/commit/c9b8e1f96953321b54b796baef731c8f72587115
    # [1] https://github.com/pypa/advisory-database/issues/207#issuecomment-2491830484
]

# Dictionary mapping benchmark names to the corresponding timeouts
BENCHMARK_TO_TIMEOUT = {
    "keyset_projection": 3,
    "keyset_cross_product_per_size": 35,
    "keyset_cross_product_per_factors": 24,
}

sm = SessionManager(
    package=PACKAGE_NAME,
    directory=CWD,
    smoketest_script=SMOKETEST_SCRIPT,
    parallel_tests=False,
    min_coverage=MIN_COVERAGE,
    audit_versions=AUDIT_VERSIONS,
    audit_suppressions=AUDIT_SUPPRESSIONS,
)

sm.build()

sm.black()
sm.isort()
sm.mypy()
sm.pylint()
sm.pydocstyle()

sm.smoketest()
sm.release_smoketest()
sm.test()
sm.test_fast()
sm.test_slow()
sm.test_doctest()

sm.docs_linkcheck()
sm.docs_doctest()
sm.docs()

sm.audit()
