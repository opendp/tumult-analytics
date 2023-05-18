"""Type checking helpers for the analytics module."""

from typing import Any

from tmlt.core.utils.exact_number import ExactNumber


def is_exact_number_tuple(obj: Any):
    """Validate that a privacy budget is of type Tuple[ExactNumber, ExactNumber]."""
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], ExactNumber)
        and isinstance(obj[1], ExactNumber)
    )
