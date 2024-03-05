"""Private utility functions."""
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024


def assert_is_identifier(identifier: str):
    """Check that the given ``identifier`` is a valid table name."""
    if not identifier.isidentifier():
        raise ValueError(
            "Names must be valid Python identifiers: they can only contain "
            "alphanumeric characters and underscores, and cannot begin with a number."
        )
