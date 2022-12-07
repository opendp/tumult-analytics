"""Cleanup functions for Analytics.

@nodoc.
"""
# pylint: disable=unused-import
import warnings

from tmlt.analytics.utils import cleanup, remove_all_temp_tables

warnings.warn(
    "The contents of the cleanup module have been moved to tmlt.analytics.utils.",
    DeprecationWarning,
    stacklevel=2,
)
