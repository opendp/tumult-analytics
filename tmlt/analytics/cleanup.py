"""Cleanup functions for Analytics."""

from tmlt.core.utils import cleanup as core_cleanup


def cleanup():
    """Cleanup the temporary table currently in use.

    If you call `spark.stop()`, you should call this function first.
    """
    core_cleanup.cleanup()


def remove_all_temp_tables():
    """Remove all temporary tables created by Analytics.

    This will remove all Analytics-created temporary tables in the current
    Spark data warehouse, whether those tables were created by the current
    Analytics session or previous Analytics sessions.
    """
    core_cleanup.remove_all_temp_tables()
