"""Private utility functions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import logging
from collections.abc import Iterable
from typing import Optional, Type

from pyspark.sql import DataFrame


def type_name(ty: type) -> str:
    """Generate a type identifier for a given type.

    Returns an identifier for the given type that can be used to unambiguously
    refer to the type, e.g. in serializers or error messages.
    """
    return f"{ty.__module__}.{ty.__qualname__}"


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


def validate_collection(
    test_collection: Iterable,
    required_elements: Iterable,
    optional_elements: Iterable,
    collection_name: str,
    elements_name: str,
    error_type: Type[Exception] = ValueError,
) -> None:
    """Validates that the provided collection contains expected elements.

    In particular, it checks that the collection contains every required element,
    and no elements that are not required or optional.

    If the collection does not pass validation, raises an exception that explains
    the problem.

    Args:
        test_collection: The collection to check.
        required_elements: The collection of elements which must all be present.
        optional_elements: The collection of elements which are allowed but not
            required. The optional and required elements must be disjoint.
        collection_name: The name of the collection to check. Used in error
            messages.
        elements_name: The name of the elements in the collection. Should be plural.
            Used in error messages.
        error_type: The type of error to raise if validation fails.

    Example:
        >>> try:
        ...   validate_collection(
        ...     test_collection={"a", "b"},
        ...     required_elements={"a", "c"},
        ...     optional_elements={"b"},
        ...     collection_name="alphabet",
        ...     elements_name="letters"
        ...   )
        ... except Exception as e:
        ...   print(f"{type(e).__name__}: {e}")
        ValueError: alphabet is missing required letters: ['c']
        Required letters: ['a', 'c']
        Optional letters: ['b']
        >>> validate_collection(
        ...   test_collection={"a", "b", "c"},
        ...   required_elements={"a", "c"},
        ...   optional_elements={"b"},
        ...   collection_name="alphabet",
        ...   elements_name="letters"
        ... )
        >>> try:
        ...    validate_collection(
        ...     test_collection={"a", "b", "c", "d"},
        ...     required_elements={"a", "c"},
        ...     optional_elements={"b"},
        ...     collection_name="alphabet",
        ...     elements_name="letters"
        ...   )
        ... except Exception as e:
        ...   print(f"{type(e).__name__}: {e}")
        ValueError: alphabet has unexpected letters: ['d']
        Required letters: ['a', 'c']
        Optional letters: ['b']
    """
    test_set = set(test_collection)
    required_set = set(required_elements)
    allowed_set = required_set | set(optional_elements)

    missing_elements = required_set - test_set
    if missing_elements:
        raise error_type(
            f"{collection_name} is missing required {elements_name}: "
            f"{sorted(missing_elements)}\n"
            f"Required {elements_name}: {sorted(required_set)}\n"
            f"Optional {elements_name}: {sorted(optional_elements)}"
        )

    unexpected_elements = test_set - allowed_set
    if unexpected_elements:
        raise error_type(
            f"{collection_name} has unexpected {elements_name}: "
            f"{sorted(unexpected_elements)}\n"
            f"Required {elements_name}: {sorted(required_set)}\n"
            f"Optional {elements_name}: {sorted(optional_elements)}"
        )


def setup_logger(logger_name: Optional[str] = None):
    """Set up the INFO logger for the analytics and set Py4J level to WARNING."""
    logger = logging.getLogger()
    if logger_name:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    class CustomFormatter(logging.Formatter):
        """Custom log handler to format the output."""

        def format(self, record):
            return record.getMessage()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # Set the log level for Py4J to WARNING to suppress INFO logs
    py4j_logger = logging.getLogger("py4j")
    py4j_logger.setLevel(logging.WARNING)

    return logger
