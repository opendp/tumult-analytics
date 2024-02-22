"""Common fixtures for Session integration tests."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import pytest

from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.protected_change import AddMaxRows, AddRowsWithID
from tmlt.analytics.session import Session

INF_BUDGET = PureDPBudget(float("inf"))
INF_BUDGET_ZCDP = RhoZCDPBudget(float("inf"))


@pytest.fixture
def session(_session_data, request):
    """A Session with some sample data.

    This fixture requires a parameter (typically passed by setting the
    `indirect` option to parametrize) specifying the privacy budget. Setting it
    up this way allows parametrizing tests to run with Sessions that use
    multiple privacy definitions without duplicating all of the test logic.
    """
    assert hasattr(
        request, "param"
    ), "The session fixture requires a parameter indicating its budget"
    budget = request.param
    assert isinstance(
        budget, PrivacyBudget
    ), "The session fixture parameter must be a PrivacyBudget"

    sess = (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_id_space("a")
        .with_id_space("b")
        # a and b use the same data, but they're still separate identifier
        # spaces; this is just to check that things like cross-ID-space joins
        # are detected.
        .with_private_dataframe(
            "id_a1", _session_data["id1"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a2", _session_data["id2"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a3", _session_data["id3"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a4", _session_data["id4"], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_b1", _session_data["id1"], protected_change=AddRowsWithID("id", "b")
        )
        .with_private_dataframe(
            "id_b2", _session_data["id1"], protected_change=AddRowsWithID("id", "b")
        )
        .with_private_dataframe(
            "rows_1", _session_data["rows1"], protected_change=AddMaxRows(2)
        )
        .build()
    )
    return sess
