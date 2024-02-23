"""Cleanup functions for Analytics.

@nodoc.
"""
# pylint: disable=unused-import
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import warnings

from tmlt.analytics.utils import cleanup, remove_all_temp_tables

warnings.warn(
    "The contents of the cleanup module have been moved to tmlt.analytics.utils.",
    DeprecationWarning,
    stacklevel=2,
)
