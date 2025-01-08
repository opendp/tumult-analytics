"""Unit tests for (v2) KeySet.detect."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, ContextManager

import pytest
from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics._keyset_v2 import KeySet, KeySetPlan


def test_detect():
    """KeySet.detect works as expected."""
    ks = KeySet._detect(["A", "B"])  # pylint: disable=protected-access
    assert isinstance(ks, KeySetPlan)
    assert ks.columns() == ["A", "B"]

    ks = KeySet._detect(["B", "A"])  # pylint: disable=protected-access
    assert isinstance(ks, KeySetPlan)
    assert ks.columns() == ["B", "A"]


@parametrize(
    Case("no_columns")(
        columns=[],
        expectation=pytest.raises(
            ValueError, match="Detect must be used on a non-empty collection of columns"
        ),
    ),
    Case("empty_column_name")(
        columns=[""],
        expectation=pytest.raises(
            ValueError, match="Empty column names are not allowed"
        ),
    ),
    Case("non_string_column_name")(
        columns=[1],
        expectation=pytest.raises(ValueError, match="Column names must be strings"),
    ),
)
def test_invalid(columns: Any, expectation: ContextManager[None]):
    """Invalid domains are rejected."""
    with expectation:
        KeySet._detect(columns)  # pylint: disable=protected-access
