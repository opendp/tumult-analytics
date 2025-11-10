"""Unit tests for (v2) KeySet.detect."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, ContextManager

import pytest
from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics import KeySet
from tmlt.analytics.keyset._keyset import KeySetPlan


def test_detect():
    """KeySet.detect works as expected."""
    ks = KeySet._detect(["A", "B"])
    assert isinstance(ks, KeySetPlan)
    assert ks.columns() == ["A", "B"]

    ks = KeySet._detect(["B", "A"])
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
        KeySet._detect(columns)
