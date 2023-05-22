"""Tests for TruncationStrategy."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import pytest

from tmlt.analytics.truncation_strategy import TruncationStrategy


@pytest.mark.parametrize("threshold", [(1), (8)])
def test_dropexcess(threshold: int):
    """Tests that DropExcess works for valid thresholds."""
    ts = TruncationStrategy.DropExcess(threshold)
    assert ts.max_records == threshold


@pytest.mark.parametrize("threshold", [(-1), (0)])
def test_invalid_dropexcess(threshold: int):
    """Tests that invalid private source errors on post-init."""
    with pytest.raises(
        ValueError, match="At least one record must be kept for each join key."
    ):
        TruncationStrategy.DropExcess(threshold)
