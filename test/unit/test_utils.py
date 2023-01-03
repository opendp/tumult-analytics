"""Unit tests for :mod:`~tmlt.analytics.utils`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
from tmlt.analytics.utils import check_installation


### Test for tmlt.analytics.utils.check_installation()
# We want the `spark` argument here so that the test will use the
# (session-wide, pytest-provided) spark session.
def test_check_installation(spark) -> None:  # pylint: disable=unused-argument
    """Test that check_installation works (doesn't raise an error)."""
    check_installation()
