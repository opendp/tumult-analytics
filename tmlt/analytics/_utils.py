"""Private utility functions."""
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from pyspark.sql import DataFrame


class AnalyticsInternalError(AssertionError):
    """Generic error to raise for internal analytics errors."""

    def __init__(self, message: str):
        """Initialize the error.

        Args:
            message: context-specific message describing the error.
        """
        common_message = (
            "\n\nThis is probably a bug! Please let us know about it at:\n"
            "https://gitlab.com/tumult-labs/analytics/-/issues/new\n"
        )
        super().__init__(message + common_message)


def assert_is_identifier(identifier: str):
    """Check that the given ``identifier`` is a valid table name."""
    if not identifier.isidentifier():
        raise ValueError(
            "Names must be valid Python identifiers: they can only contain "
            "alphanumeric characters and underscores, and cannot begin with a number."
        )


def dataframe_is_empty(df: DataFrame) -> bool:
    """Checks if a pyspark dataframe is empty.

    Will use the more efficient DataFrame.isEmpty() method if it's available
    (i.e. if pySpark > 3.3.0).
    """
    isEmpty = getattr(df, "isEmpty()", None)
    if callable(isEmpty):
        return df.isEmpty()

    return df.count() == 0
