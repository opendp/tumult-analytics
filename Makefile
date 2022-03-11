SHELL = /bin/bash
# This causes all targets to execute their entire script within a single shell,
# as opposed to using a subshell per line. See
# https://www.gnu.org/software/make/manual/html_node/One-Shell.html.
.ONESHELL:

define targets
lint
lint-fix
test
test-fast
test-slow
test-doctest
test-examples
benchmark
package
docs-fast
endef

define make-target
.PHONY: $(1)
$(1):
	@export BASE_DIR="$(CURDIR)"
	export PACKAGE_DIR=""
	source ./.buildscripts.base
	[ -f ./.buildscripts ] && source ./.buildscripts
	$(1)
	echo
endef

$(foreach target, $(targets), $(eval $(call make-target,$(target))))

################################################################################
# Cleanup
#
# The above scripts (especially tests) generate a bunch of junk in the
# repository that isn't generally useful to keep around. This helps clean all of
# those files/directories up.
################################################################################

define clean-files
**/*.pyc
junit.xml
coverage.xml
.coverage
**/*.nbconvert.ipynb
dist/
public/
spark-warehouse/
endef

.PHONY: clean
clean:
	@git clean -X -n -- $(foreach f, $(clean-files),'$(f)')
	read -p "Cleaning would remove the above files. Continue? [y/N] " CLEAN
	if [[ "$$CLEAN" = "y" || "$$CLEAN" = "yes" ]]; then
		git clean -X -f -- $(foreach f, $(clean-files),'$(f)')
	fi
