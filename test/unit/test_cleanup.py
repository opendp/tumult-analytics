"""Tests for Analytics cleanup functions."""


from unittest.mock import patch

from tmlt.analytics.utils import cleanup, remove_all_temp_tables


def test_cleanup() -> None:
    """Test Analytics cleanup function."""
    with patch("tmlt.analytics.utils.core_cleanup.cleanup") as mock_core_cleanup:
        cleanup()
        mock_core_cleanup.assert_called_once()


def test_remove_all_temp_tables() -> None:
    """Test Analytics remove_all_temp_tables function."""
    with patch(
        "tmlt.analytics.utils.core_cleanup.remove_all_temp_tables"
    ) as mock_core_remove:
        remove_all_temp_tables()
        mock_core_remove.assert_called_once()
