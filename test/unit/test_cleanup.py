"""Tests for Analytics cleanup functions."""

from unittest import TestCase
from unittest.mock import patch

from tmlt.analytics.cleanup import cleanup, remove_all_temp_tables


class TestCleanup(TestCase):
    """Tests for tmlt.analytics.cleanup functions."""

    @patch("tmlt.analytics.cleanup.core_cleanup.cleanup")
    def test_cleanup(self, mock_core_cleanup) -> None:  # pylint: disable=no-self-use
        """Test Analytics cleanup function."""
        cleanup()
        mock_core_cleanup.assert_called_once()

    @patch("tmlt.analytics.cleanup.core_cleanup.remove_all_temp_tables")
    # pylint: disable=no-self-use
    def test_remove_all_temp_tables(self, mock_core_remove) -> None:
        """Test Analytics remove_all_temp_tables function."""
        remove_all_temp_tables()
        mock_core_remove.assert_called_once()

    # pylint: enable=no-self-use
