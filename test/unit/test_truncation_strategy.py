"""Tests for TruncationStrategy."""

# <placeholder: boilerplate>

import unittest

from parameterized import parameterized

from tmlt.analytics.truncation_strategy import TruncationStrategy


class TestAttributes(unittest.TestCase):
    """Tests valid and invalid attributes on TruncationStrategy variants."""

    @parameterized.expand([(1,), (8,)])
    def test_dropexcess(self, threshold: int):
        """Tests that DropExcess works for valid thresholds."""
        ts = TruncationStrategy.DropExcess(threshold)
        self.assertEqual(ts.max_records, threshold)

    @parameterized.expand([(-1,), (0,)])
    def test_invalid_dropexcess(self, threshold: int):
        """Tests that invalid private source errors on post-init."""
        with self.assertRaisesRegex(
            ValueError, "At least one record must be kept for each join key."
        ):
            TruncationStrategy.DropExcess(threshold)
