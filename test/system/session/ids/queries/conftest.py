"""Testing utilities."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Dict, Optional, Tuple

from pyspark.sql import DataFrame

from tmlt.analytics import AddRowsWithID, PrivacyBudget, ProtectedChange, Session


def make_session(
    budget: PrivacyBudget,
    private_tables: Dict[str, Tuple[DataFrame, ProtectedChange]],
    public_tables: Optional[Dict[str, DataFrame]] = None,
) -> Session:
    """Shorthand for building a Session with a budget and a collection of tables."""
    id_spaces = set()
    builder = Session.Builder().with_privacy_budget(budget)
    for t in private_tables:
        df, pc = private_tables[t]
        builder = builder.with_private_dataframe(t, df, pc)
        if isinstance(pc, AddRowsWithID):
            id_spaces.add(pc.id_space)
    for id_space in id_spaces:
        builder = builder.with_id_space(id_space)
    if public_tables:
        for t in public_tables:
            builder = builder.with_public_dataframe(t, public_tables[t])
    return builder.build()
