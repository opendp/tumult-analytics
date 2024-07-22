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

lint:
	nox --no-venv -t lint

test:
	nox --no-venv -s test
test-fast:
	nox --no-venv -s test_fast
test-slow:
	nox --no-venv -s test_slow
test-doctest:
	nox --no-venv -s test_doctest
test-examples:
	nox --no-venv -s test_examples

docs:
	nox --no-venv -s docs
docs-linkcheck:
	nox --no-venv -s docs_linkcheck
docs-doctest:
	nox --no-venv -s docs_doctest

package:
	nox --no-venv -s build


# The above scripts (especially tests) generate a bunch of junk in the
# repository that isn't generally useful to keep around. This helps clean all of
# those files/directories up.

define clean-files
tmlt/**/__pycache__/
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
tmlt/**/spark-warehouse/
test/**/spark-warehouse/
examples/spark-warehouse/
.mypy_cache/
endef

.PHONY: clean
clean:
	@git clean -x -n -- $(foreach f, $(clean-files),'$(f)')
	read -p "Cleaning would remove the above files. Continue? [y/N] " CLEAN
	if [[ "$$CLEAN" = "y" || "$$CLEAN" = "yes" ]]; then
	  git clean -x -f -- $(foreach f, $(clean-files),'$(f)')
	fi


