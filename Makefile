# This makefile provides shorthands for various operations that are common
# during development. When it invokes nox, it skips virtual environment creation
# to give faster results. However, in some corner cases this may cause
# inconsistencies with the CI; if this is a problem, run nox manually without
# the --no-venv option.

SHELL = /bin/bash

.PHONY: lint test docs package

# This causes all targets to execute their entire script within a single shell,
# as opposed to using a subshell per line. See
# https://www.gnu.org/software/make/manual/html_node/One-Shell.html.
.ONESHELL:

# Run all linters
lint:
	uv run nox --no-venv -t lint

# Run all tests
test:
	uv run nox --no-venv -s smoketest test-fast test-slow test-doctest docs-doctest

# Builds the docs and checks links
docs:
	uv run nox --no-venv -s docs docs-linkcheck

# Builds the source distribution and wheels
package:
	uv run nox --no-venv -s build

# The scripts generate a bunch of junk in the repository that isn't generally
# useful to keep around. This cleans up all of those files/directories.

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
