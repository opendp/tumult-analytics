"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.

Most sessions in this file are designed to work either directly in a development
environment (i.e. with nox's --no-venv option) or in a nox-managed virtualenv
(as they would run in the CI). Sessions that only work in one or the other will
indicate this in their docstrings.
"""
# TODO(#2140): Once support for is added to nox-poetry (see
#   https://github.com/cjolowicz/nox-poetry/issues/663), some of the
#   installation lists here can be rewritten in terms of dependency groups,
#   making the pyproject file more of a single source for information about
#   dependencies.

import datetime
import os
import re
import subprocess
import tempfile
from functools import wraps
from glob import glob
from pathlib import Path
from typing import Dict, List

# The distinction between these two session types is that poetry_session
# automatically limits installations to the version numbers in the Poetry lock
# file, while nox_session does not. Otherwise, their interfaces should be
# identical.
import nox
from nox import session as nox_session
from nox_poetry import session as poetry_session

#### Project-specific settings ####

PACKAGE_NAME = "tmlt.analytics"
"""Name of the package."""
PACKAGE_SOURCE_DIR = "tmlt/analytics"
"""Relative path from the project root to its source code."""
# TODO(#2177): Once we have a better way to self-test our code, use it here in
#              place of this import check.
SMOKETEST_SCRIPT = """
from tmlt.analytics.utils import check_installation
check_installation()
"""
"""Python script to run as a quick self-test."""

MIN_COVERAGE = 75
"""For test suites where we track coverage (i.e. the fast tests and the full
test suite), fail if test coverage falls below this percentage."""


# This note about credentials for dependency URLs is not currently relevant, but
# if it becomes a problem _install_overrides can be modified to write
# dependencies out to a requirements file and install from there.


# To the greatest extent possible, avoid making project-specific modifications
# to the rest of this file, except the project-specific sessions at the
# end. Make sure any changes made there are propagated to other projects that
# use this same file.

#### Additional settings ####

CWD = Path(".").resolve()
CODE_DIRS = [
    str(p) for p in [Path(PACKAGE_SOURCE_DIR).resolve(), Path("test").resolve()]
]
IN_CI = bool(os.environ.get("CI"))
PACKAGE_VERSION = subprocess.run(
    ["poetry", "version", "-s"], capture_output=True
).stdout.decode("utf-8").strip()
"""The current full package version, according to Poetry."""

#### Utility functions ####

def _install_overrides(session):
    """Handles overriding dependency versions."""
    # Install Core from dist/, if it exists there
    if os.environ.get("PARENT_PIPELINE_ID"):
        core_wheels = glob(r"./dist/*tmlt_core*-cp37*")
        if len(core_wheels) == 0:
            raise AssertionError("Expected a core wheel since PARENT_PIPELINE_ID was set "
                f"(to {os.environ.get('PARENT_PIPELINE_ID')}), but didn't find any. "
                f"There should be one in dist/, which contains: {glob(r'dist/*')}, "
            )
        # Poetry is going to expect, and require, Core version X.Y.Z (ex. "0.6.2"),
        # but the Gitlab-built Core will have a version number
        # X.Y.Z-<some other stuff>-<git commit hash>
        # (ex. "0.6.2-post11+ea346f3")
        # This overrides Poetry's dependencies with our own
        session.poetry.session.install(core_wheels[0])

def install(*decorator_args, **decorator_kwargs):
    """Install packages into the test virtual environment.

    Installs one or more given packages, if the current environment supports
    installing packages. Parameters to the decorator are passed directly to
    nox's session.install, so anything valid there can be passed to the
    decorator.

    The difference between using this decorator and using a normal
    session.install call is that this decorator will automatically skip
    installation when nox is not running tests in a virtual environment, rather
    than raising an error. This is helpful for writing sessions that can be used
    either in sandboxed environments in the CI or directly in developers'
    working environments.
    """
    def decorator(f):
        @wraps(f)
        def inner(session, *args, **kwargs):
            if session.virtualenv.is_sandboxed:
                session.install(*decorator_args, **decorator_kwargs)
            else:
                session.log("Skipping package installation, non-sandboxed environment")
            return f(session, *args, **kwargs)
        return inner
    return decorator

def install_package(f):
    """Install the main package a dev wheel into the test virtual environment.

    Installs the package from this repository and all its dependencies, if the
    current environment supports installing packages. If wheels for
    the current dev version (from `poetry version`) are not already present in
    `dist/`, this will build them.

    Similar to the @install() decorator, this decorator automatically skips
    installation in non-sandboxed environments.
    """
    @wraps(f)
    def inner(session, *args, **kwargs):
        if session.virtualenv.is_sandboxed:
            temp_dir = session.create_tmp()
            out = session.run(
                "pip",
                "download",
                f"{PACKAGE_NAME}=={PACKAGE_VERSION}",
                "--find-links",
                f"{CWD}/dist/",
                "--only-binary",
                PACKAGE_NAME,
                "-d",
                temp_dir,
                "--no-deps",
                silent=True,
                success_codes=[0, 1],
            )
            if "No matching distribution" in out:
                build(session)

            session.install(
                f"{PACKAGE_NAME}=={PACKAGE_VERSION}",
                "--find-links", f"{CWD}/dist/", "--only-binary", PACKAGE_NAME
            )
            _install_overrides(session)
        else:
            session.log("Skipping package installation, non-sandboxed environment")
        return f(session, *args, **kwargs)
    return inner

def show_installed(f):
    """Show a list of installed packages in the active environment for debugging.

    By default, the package list is only shown when running in the CI, as that
    is where it is most difficult to debug. However, the show_installed option
    can be passed to any function with this decorator to force showing or not
    showing it.
    """
    @wraps(f)
    def inner(session, *args, show_installed: bool = None, **kwargs):
        show_installed = show_installed if show_installed is not None else IN_CI
        if show_installed:
            session.run("pip", "freeze")
        return f(session, *args, **kwargs)
    return inner

def with_clean_workdir(f):
    """If in a sandboxed virtualenv, execute session from an empty tempdir.

    This decorator works around an issue with the tests where they will try to
    use the code (and thus the shared libraries) from the repository rather than
    the wheel that should be used. By moving to a temporary directory before
    running the tests, the repository is not in the Python load path, so the
    problem is resolved.
    """
    @wraps(f)
    def inner(session, *args, **kwargs):
        if session.virtualenv.is_sandboxed:
            with tempfile.TemporaryDirectory() as workdir, session.cd(workdir):
                return f(session, *args, **kwargs)
        else:
            return f(session, *args, **kwargs)
    return inner

#### Linting ####

# Some testing-related packages need to be installed for linting because they
# are imported in the tests, and so are required for some of the linters to work
# correctly.

@poetry_session(tags=["lint"], python="3.7")
@install_package
@install("black")
@show_installed
def black(session):
    """Run black. If the --check argument is given, only check, don't make changes."""
    check_flags = ["--check", "--diff"] if "--check" in session.posargs else []
    session.run(
        "black", "--skip-magic-trailing-comma",
        *check_flags, *CODE_DIRS
    )

@poetry_session(tags=["lint"], python="3.7")
@install_package
@install("isort[pyproject]", "pytest")
@show_installed
def isort(session):
    """Run isort. If the --check argument is given, only check, don't make changes."""
    check_flags = ["--check-only", "--diff"] if "--check" in session.posargs else []
    session.run("isort", *check_flags, *CODE_DIRS)

@poetry_session(tags=["lint"], python="3.7")
@install_package
@install("mypy")
@show_installed
def mypy(session):
    """Run mypy."""
    session.run("mypy", *CODE_DIRS)

@poetry_session(tags=["lint"], python="3.7")
@install_package
@install("pylint", "pytest")
@show_installed
def pylint(session):
    """Run pylint."""
    session.run("pylint", "--score=no", *CODE_DIRS)

@poetry_session(tags=["lint"], python="3.7")
@install_package
@install("pydocstyle[toml]")
@show_installed
def pydocstyle(session):
    """Run pydocstyle."""
    session.run("pydocstyle", *CODE_DIRS)

#### Tests ####

@install_package
@install("pytest", "coverage")
@show_installed
@with_clean_workdir
def _test(
    session,
    test_dirs: List[str] = None,
    min_coverage: int = MIN_COVERAGE,
    extra_args: List[str] = None,
):
    test_paths = test_dirs or CODE_DIRS
    extra_args = extra_args or []
    # If the user passes args, pass them on to pytest. The main reason this is
    # useful is for specifying a particular subset of tests to run, so clear
    # test_paths to allow that use case.
    if session.posargs:
        test_paths = []
        extra_args.extend(session.posargs)

    test_options = [
        "-r fEs", "--verbose", "--disable-warnings", f"--junitxml={CWD}/junit.xml",
        # Show runtimes of the 10 slowest tests, for later comparison if needed.
        "--durations=10",
        *extra_args,
        *[str(p) for p in test_paths],
    ]
    session.run("coverage", "run", "--branch", "-m", "pytest", *test_options)
    session.run("coverage", "html", f"--include={CWD}/{PACKAGE_SOURCE_DIR}/*", f"--directory={CWD}/coverage/")
    session.run("coverage", "report", f"--include={CWD}/{PACKAGE_SOURCE_DIR}/*", f"--fail-under={min_coverage}")

@install_package
@show_installed
@with_clean_workdir
def _smoketest(session):
    """Run a no-extra-dependencies smoketest on the package."""
    session.run("python", "-c", SMOKETEST_SCRIPT)

# Only this session, test_doctest, and test_examples one get the 'test' tag,
# because the others are just subsets of this session so there's no need to run
# them again.
@poetry_session(tags=["test"], python="3.7")
def test(session):
    """Run all tests."""
    _test(session)

@poetry_session(python="3.7")
def test_fast(session):
    """Run tests without the slow attribute."""
    _test(session, extra_args=["-m", "not slow"])

@poetry_session(python="3.7")
def test_slow(session):
    """Run tests with the slow attribute."""
    _test(session, min_coverage=0, extra_args=['-m', 'slow'])

@poetry_session(tags=["test"], python="3.7")
def test_doctest(session):
    """Run doctest on code examples in docstrings."""
    _test(
        session, test_dirs=[Path(PACKAGE_SOURCE_DIR).resolve()],
        min_coverage=0, extra_args=["--doctest-modules"]
    )

@poetry_session(tags=["test"])
def test_smoketest(session):
    """Smoke test a wheel as it would be installed on a user's machine."""
    _smoketest(session)

@poetry_session(tags=["test"], python="3.7")
@install_package
@install("notebook", "nbconvert", "matplotlib", "seaborn")
@show_installed
def test_examples(session):
    """Run all examples."""
    examples_path = CWD / "examples"
    if not examples_path.exists():
        session.error("No examples directory found, nothing to run")
    examples_py = []
    examples_ipynb = []
    unknown = []
    ignored = []
    for f in examples_path.iterdir():
        if f.is_file and f.suffix == ".py":
            examples_py.append(f)
        elif f.is_file and f.suffix == ".ipynb":
            if ".nbconvert" not in f.suffixes:
                examples_ipynb.append(f)
            else:
                ignored.append(f)
        else:
            unknown.append(f)
    for py in examples_py:
        session.run("python", str(py))
    for nb in examples_ipynb:
        session.run("jupyter", "nbconvert", "--to=notebook", "--execute", str(nb))
    if ignored:
        session.log(
            f"Ignored: {', '.join(str(f) for f in ignored)}"
        )
    if unknown:
        session.warn(
            f"Found unknown files in examples: {', '.join(str(f) for f in unknown)}"
        )

### Test various dependency configurations ###
# Test each with oldest and newest allowable deps. Typeguard and typing-extensions
# excluded because all of the allowed versions in pyproject.toml claim support
# for all allowable python versions.

@nox_session
@install("pytest", "coverage")
@with_clean_workdir
@nox.parametrize(
    "python,pyspark,sympy,pandas,core",
[
    ("3.7", "3.0.0", "1.8", "1.2.0", "==0.11.5"),
    ("3.7", "3.1.1", "1.9", "1.3.5", ">=0.12.0,<0.13.0"),
    ("3.7", "3.2.0", "1.9", "1.3.5", ">=0.12.0,<0.13.0"),
    ("3.7", "3.3.3", "1.9", "1.3.5", ">=0.12.0,<0.13.0"),
    ("3.8", "3.0.0", "1.8", "1.2.0", "==0.11.5"),
    ("3.8", "3.5.0", "1.9", "1.5.3", ">=0.12.0,<0.13.0"),
    ("3.9", "3.0.0", "1.8", "1.2.0", "==0.11.5"),
    ("3.9", "3.5.0", "1.9", "1.5.3", ">=0.12.0,<0.13.0"),
    ("3.10", "3.0.0", "1.8", "1.4.0", "==0.11.5"),
    ("3.10", "3.5.0", "1.9", "1.5.3", ">=0.12.0,<0.13.0"),
    ("3.11", "3.4.0", "1.8", "1.5.0", "==0.11.5"),
    ("3.11", "3.5.0", "1.9", "1.5.3", ">=0.12.0,<0.13.0"),
],
ids=[
    "3.7-oldest",
    "3.7-pyspark3.1",
    "3.7-pyspark3.2",
    "3.7-newest",
    "3.8-oldest",
    "3.8-newest",
    "3.9-oldest",
    "3.9-newest",
    "3.10-oldest",
    "3.10-newest",
    "3.11-oldest",
    "3.11-newest",
])
def test_multi_deps(session, pyspark, sympy, pandas, core):
    """Run tests using various dependencies."""
    session.install(
                f"{PACKAGE_NAME}=={PACKAGE_VERSION}",
                "--find-links", f"{CWD}/dist/", "--only-binary", PACKAGE_NAME
            )
    session.install("tmlt.core>0.5.0")
    session.install(f"pyspark=={pyspark}", f"sympy=={sympy}", f"pandas=={pandas}",
    f"tmlt.core{core}")
    session.run("pip", "freeze")
    test_options = [
        "-rfs", "--disable-warnings", f"--junitxml={CWD}/junit.xml",
        # Show runtimes of the 10 slowest tests, for later comparison if needed.
        "--durations=10",
        "-m", "not slow",
        *[str(p) for p in CODE_DIRS],
    ]
    session.run("coverage", "run", "--branch", "-m", "pytest", *test_options)
    session.run("coverage", "html", f"--include={CWD}/{PACKAGE_SOURCE_DIR}/*",
     f"--directory={CWD}/coverage/")
    session.run("coverage", "report", f"--include={CWD}/{PACKAGE_SOURCE_DIR}/*",
     "--fail-under=75")

#### Documentation ####

@install_package
@install(
    "pandoc", "pydata-sphinx-theme", "scanpydoc", "sphinx",
    "sphinx-autoapi", "sphinx-autodoc-typehints", "sphinx-copybutton",
    "sphinx-panels", "sphinxcontrib-bibtex", "sphinxcontrib-images",
)
@show_installed
def _run_sphinx(session, builder: str):
    sphinx_options = ["-n", "-W", "--keep-going"]
    session.run("sphinx-build", "doc/", "public/", f"-b={builder}", *sphinx_options)

@poetry_session(tags=["docs"], python="3.7")
def docs_linkcheck(session):
    """Run linkcheck on docs."""
    _run_sphinx(session, "linkcheck")

@poetry_session(tags=["docs"], python="3.7")
@install("matplotlib", "seaborn")
def docs_doctest(session):
    """Run doctest on code examples in documentation."""
    _run_sphinx(session, "doctest")

@poetry_session(tags=["docs"], python="3.7")
def docs(session):
    """Generation HTML documentation."""
    _run_sphinx(session, "html")

#### Release management ####

@nox_session(python=None)
def prepare_release(session):
    """Update files in preparation for a release.

    The version number for the new release should be in the VERSION environment
    variable.
    """
    version = os.environ.get("VERSION")
    if not version:
        session.error("VERSION not set, unable to prepare release")

    # Check version number against our allowed version format. This matches a
    # subset of semantic versions that closely matches PEP440 versions. Some
    # examples include: 0.1.2, 1.2.3-alpha.2, 1.3.0-rc.1
    version_regex = (
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-(alpha|beta|rc)\.(0|[1-9]\d*))?$"
    )
    if not re.match(version_regex, version):
        session.error(f"VERSION {version} is not a valid version number.")
    session.debug(f"Preparing release {version}")

    # Replace "Unreleased" section header in changelog for non-prerelease
    # releases. Between the base version and prerelease number is the only place
    # a hyphen can appear in the version number, so just checking for that
    # indicates whether a version is a prerelease.
    is_pre_release = "-" in version
    if not is_pre_release:
        session.log("Updating CHANGELOG.rst unreleased version...")
        with Path("CHANGELOG.rst").open("r") as fp:
            changelog_content = fp.readlines()
        for i in range(len(changelog_content)):
            if re.match('^Unreleased$', changelog_content[i]):
                # BEFORE
                # Unreleased
                # ----------

                # AFTER
                # 1.2.3 - 2020-01-01
                # ------------------
                version_header = f'{version} - {datetime.date.today()}'
                changelog_content[i] = version_header + "\n"
                changelog_content[i + 1] = "-" * len(version_header) + "\n"
                break
        else:
            session.error(
                "Renaming unreleased section in changelog failed, "
                "unable to find matching line"
            )
        with Path("CHANGELOG.rst").open("w") as fp:
            fp.writelines(changelog_content)
    else:
        session.log("Prerelease, skipping CHANGELOG.rst update...")


@nox_session(python=None)
def post_release(session):
    """Update files after a release."""
    version_and_date_regex = (
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-(alpha|beta|rc)\.(0|[1-9]\d*))?"
        r" - \d{4}-\d{2}-\d{2}$"
    )
    # Find the latest release
    with Path("CHANGELOG.rst").open("r") as fp:
        changelog_content = fp.readlines()
        for i in range(len(changelog_content)):
            if re.match(version_and_date_regex, changelog_content[i]):
                version = changelog_content[i].split(" - ")[0]
                is_pre_release = "-" in version
                if not is_pre_release:
                    # BEFORE
                    # 1.2.3 - 2020-01-01
                    # ------------------

                    # AFTER
                    # Unreleased
                    # ----------
                    #
                    # 1.2.3 - 2020-01-01
                    # ------------------
                    new_lines = ["Unreleased\n", "----------\n", "\n"]
                    for new_line in reversed(new_lines):
                        changelog_content.insert(i, new_line)
                    break
                else:
                    session.log("Prerelease, skipping CHANGELOG.rst update...")
                    return
        else:
            session.error("Unable to find latest release in CHANGELOG.rst")
        with Path("CHANGELOG.rst").open("w") as fp:
            fp.writelines(changelog_content)



@nox_session()
def release_smoketest(session):
    """Smoke test a wheel as it would be installed on a user's machine.

    This session installs a built wheel as the user would install it, without
    Poetry, then runs a short test to ensure that the library plausibly works.

    Note: This session doesn't do anything useful when run with the `--no-venv`
          option, as it requires a clean environment to install things in.
    """
    _smoketest(session)

@nox_session()
def release_test(session):
    """Test a wheel as it would be installed on a user's machine.

    This session is used to verify that built wheels install correctly as a user
    would install them, without Poetry. It installs a wheel given as a
    positional argument, then runs the fast tests on it.

    Note: This session doesn't do anything useful when run with the `--no-venv`
          option, as it requires a clean environment to install things in.
    """
    _test(session, extra_args=["-m", "not slow"], show_installed=True)


#### Project-specific sessions ####

@poetry_session()
def build(session):
    """Build packages for distribution."""
    session.run("poetry", "build", external=True)
