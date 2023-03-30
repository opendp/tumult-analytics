"""Defines a visitor for propagating constraints through transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import List, Optional, Set, Union

from pyspark.sql import DataFrame

from tmlt.analytics.constraints import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
)
from tmlt.analytics.query_expr import (
    DropInfinity,
    DropNullAndNan,
    Filter,
    FlatMap,
    Map,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
)


def propagate_unmodified(
    _expr: Union[Filter, DropInfinity, DropNullAndNan], constraints: List[Constraint]
) -> List[Constraint]:
    """Propagate a list of constraints through a transformation unmodified."""
    # Filters, DropNullandNans and DropInfinities can only modify a table by dropping
    # rows, and all current constraints remain valid when rows are dropped, so just
    # return the input constraints unmodified.
    return constraints


def propagate_replace(
    expr: Union[ReplaceInfinity, ReplaceNullAndNan], constraints: List[Constraint]
) -> List[Constraint]:
    """Propagate a list of constraints through replacement transformations."""
    # Replacements on grouping columns potentially invalidate the MaxRowsPerGroupPerID
    # constraint, so they get dropped. All other constraints remain valid.
    constraint_propagatable_p = lambda c: (
        isinstance(c, (MaxRowsPerID, MaxGroupsPerID))
        or (
            isinstance(c, MaxRowsPerGroupPerID)
            and c.grouping_column not in expr.replace_with
        )
    )

    return [c for c in constraints if constraint_propagatable_p(c)]


def propagate_rename(expr: Rename, constraints: List[Constraint]) -> List[Constraint]:
    """Propagate a list of constraints through a Rename transformation."""

    def renamed_constraint(c: Constraint) -> Optional[Constraint]:
        if isinstance(c, MaxRowsPerID):
            return c
        elif isinstance(c, MaxGroupsPerID):
            return MaxGroupsPerID(
                expr.column_mapper.get(c.grouping_column, c.grouping_column), c.max
            )
        elif isinstance(c, MaxRowsPerGroupPerID):
            return MaxRowsPerGroupPerID(
                expr.column_mapper.get(c.grouping_column, c.grouping_column), c.max
            )
        return None

    return [c for c in map(renamed_constraint, constraints) if c is not None]


def propagate_select(expr: Select, constraints: List[Constraint]) -> List[Constraint]:
    """Propagate a list of constraints through a Select transformation."""
    dropped_p = lambda c: (
        isinstance(c, (MaxGroupsPerID, MaxRowsPerGroupPerID))
        and c.grouping_column not in expr.columns
    )
    return [c for c in constraints if not dropped_p(c)]


def propagate_map(expr: Map, constraints: List[Constraint]) -> List[Constraint]:
    """Propagate a list of constraints through a Map transformation."""
    if not expr.augment and constraints:
        raise AssertionError(
            "Non-augmenting map applied to table with constraints. "
            "This is probably a bug; please let us know about it so we can fix it!"
        )
    # Map can only add columns to the existing rows, so existing constraints
    # remain valid.
    return constraints


def propagate_flat_map(
    expr: FlatMap, constraints: List[Constraint]
) -> List[Constraint]:
    """Propagate a list of constraints through a FlatMap transformation."""
    if not expr.augment and constraints:
        raise AssertionError(
            "Non-augmenting flat map applied to table with constraints. "
            "This is probably a bug; please let us know about it so we can fix it!"
        )
    # Because rows can be duplicated arbitrarily many times by flat maps,
    # MaxRowsPerID and MaxRowsPerGroupPerID constraints cannot be propagated
    # through them; MaxGroupsPerID remains valid when existing rows are
    # duplicated, because this can't add any new groups, so they remain valid.
    return [c for c in constraints if isinstance(c, MaxGroupsPerID)]


def _propagate_join_by_stability(
    c: Constraint,
    join_stability: Optional[int],
    overlapping_cols: Set[str],
    suffix: str,
):
    """Propagate a constraint through a join based on the maximum join stability.

    This function returns a constraint that results from propagating the given
    constraint through a join with the given join stability. If that stability
    is None, the join is assumed to be unbounded. Column renaming because of
    overlapping columns is handled, appending the given suffix to any relevant
    columns mentioned in the constraint.
    """

    def col_name(base: str):
        return base + suffix if base in overlapping_cols else base

    if isinstance(c, MaxRowsPerID):
        if join_stability:
            return MaxRowsPerID(c.max * join_stability)
        else:
            return None
    elif isinstance(c, MaxGroupsPerID):
        return MaxGroupsPerID(col_name(c.grouping_column), c.max)
    elif isinstance(c, MaxRowsPerGroupPerID):
        if join_stability:
            return MaxRowsPerGroupPerID(
                col_name(c.grouping_column), c.max * join_stability
            )
        else:
            return None
    return None


def propagate_join_private(
    join_cols: Set[str],
    overlapping_cols: Set[str],
    left_constraints: List[Constraint],
    right_constraints: List[Constraint],
) -> List[Constraint]:
    """Propagate a list of constraints through a JoinPrivate transformation."""

    def max_join_stability(cs: List[Constraint]) -> Optional[int]:
        stabilities = []
        # A MaxRowsPerID constraint limits the duplication factor from a table,
        # as it limits the number of times each value in the ID column can
        # appear.
        max_rows_per_group = next((c for c in cs if isinstance(c, MaxRowsPerID)), None)
        if max_rows_per_group:
            stabilities.append(max_rows_per_group.max)
        # When a MaxGroupsPerID constraint has a grouping column in the join
        # columns, that also limits the duplication factor because each (ID,
        # grouping column) value pair can only appear a limited number of times.
        for c in [c for c in cs if isinstance(c, MaxGroupsPerID)]:
            if c.grouping_column in join_cols:
                stabilities.append(c.max)

        if not stabilities:
            return None
        return min(stabilities)

    left_stability = max_join_stability(left_constraints)
    right_stability = max_join_stability(right_constraints)

    left_propagated = [
        _propagate_join_by_stability(
            c, right_stability, overlapping_cols, suffix="_left"
        )
        for c in left_constraints
    ]
    right_propagated = [
        _propagate_join_by_stability(
            c, left_stability, overlapping_cols, suffix="_right"
        )
        for c in right_constraints
    ]
    return [c for c in left_propagated + right_propagated if c is not None]


def propagate_join_public(
    join_cols: Set[str],
    overlapping_cols: Set[str],
    public_df: DataFrame,
    constraints: List[Constraint],
) -> List[Constraint]:
    """Propagate a list of constraints through a JoinPublic transformation."""
    join_stability = max(
        public_df.select(*join_cols)
        .groupby(*join_cols)
        .count()
        .select("count")
        .toPandas()["count"]
        .to_list(),
        default=0,
    )
    propagated = [
        _propagate_join_by_stability(
            c, join_stability, overlapping_cols, suffix="_left"
        )
        for c in constraints
    ]
    return [c for c in propagated if c is not None]
