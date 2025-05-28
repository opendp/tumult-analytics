# This makefile provides shorthands for various operations that are common
# during development. When it invokes nox, it skips virtual environment creation
# to give faster results. However, in some corner cases this may cause
# inconsistencies with the CI; if this is a problem, run nox manually without
# the --no-venv option.

SHELL = /bin/bash

.PHONY: lint test test-fast test-slow test-doctest test-examples \
        docs docs-linkcheck docs-doctest package

# This causes all targets to execute their entire script within a single shell,
# as opposed to using a subshell per line. See
# https://www.gnu.org/software/make/manual/html_node/One-Shell.html.
.ONESHELL:

# Run all linters
lint:
	poetry run nox --no-venv -t lint

# Run all tests
test:
	poetry run nox --no-venv -s test

# Run only fast tests
test-fast:
	poetry run nox --no-venv -s test_fast

# Run only slow tests
test-slow:
	poetry run nox --no-venv -s test_slow

# Run code examples in docstrings
test-doctest:
	poetry run nox --no-venv -s test_doctest

# Build the docs
docs:
	poetry run nox --no-venv -s docs

# Check that none of the links in the documentation return 404 errors
docs-linkcheck:
	poetry run nox --no-venv -s docs_linkcheck

# Run code examples in Sphinx documentation
docs-doctest:
	poetry run nox --no-venv -s docs_doctest

# Builds 
package:
	poetry run nox --no-venv -s build


# The above scripts (especially tests) generate a bunch of junk in the
# repository that isn't generally useful to keep around. This helps clean all of
# those files/directories up.

define clean-files
src/**/__pycache__/
test/**/__pycache__/
junit.xml
coverage.xml
.coverage
coverage/
benchmark_output/
**/*.nbconvert.ipynb
dist/
public/
spark-warehouse/
src/**/spark-warehouse/
test/**/spark-warehouse/
examples/spark-warehouse/
.mypy_cache/
doc/reference/api
endef

.PHONY: clean
clean:
	@git clean -x -n -- $(foreach f, $(clean-files),'$(f)')
	read -p "Cleaning would remove the above files. Continue? [y/N] " CLEAN
	if [[ "$$CLEAN" = "y" || "$$CLEAN" = "yes" ]]; then
	  git clean -x -f -- $(foreach f, $(clean-files),'$(f)')
	fi


