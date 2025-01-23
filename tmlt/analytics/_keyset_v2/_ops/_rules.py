"""Rules for rewriting KeySetOp trees."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from collections import deque
from functools import reduce, wraps
from typing import Callable, cast

from tmlt.analytics import AnalyticsInternalError

from ._base import KeySetOp
from ._cross_join import CrossJoin, InMemoryCrossJoin
from ._detect import Detect
from ._filter import Filter
from ._from_dataframe import FromSparkDataFrame
from ._from_tuples import FromTuples
from ._join import Join
from ._project import Project
from ._subtract import Subtract

_IN_MEMORY_CROSS_JOIN_THRESHOLD = 2**20
"""The maximum number of output rows where an InMemoryCrossJoin will be applied."""


def depth_first(func: Callable[[KeySetOp], KeySetOp]) -> Callable[[KeySetOp], KeySetOp]:
    """Recursively apply the given method to an op-tree, depth-first."""

    @wraps(func)
    def wrapped(op: KeySetOp) -> KeySetOp:
        if isinstance(op, (Detect, FromTuples, FromSparkDataFrame)):
            return func(op)
        elif isinstance(op, CrossJoin):
            return func(type(op)(tuple(wrapped(f) for f in op.factors)))
        elif isinstance(op, Join):
            left = wrapped(op.left)
            right = wrapped(op.right)
            return func(Join(left, right))
        elif isinstance(op, Project):
            child = wrapped(op.child)
            return func(Project(child, op.projected_columns))
        elif isinstance(op, Filter):
            child = wrapped(op.child)
            return func(Filter(child, op.condition))
        elif isinstance(op, Subtract):
            left = wrapped(op.left)
            right = wrapped(op.right)
            return func(Subtract(left, right))
        else:
            raise AnalyticsInternalError(
                f"Unhandled KeySetOp subtype {type(op).__qualname__} encountered."
            )

    return wrapped


def breadth_first(
    func: Callable[[KeySetOp], KeySetOp]
) -> Callable[[KeySetOp], KeySetOp]:
    """Recursively apply the given method to an op-tree, breadth-first.

    "Breadth-first" is a bit fuzzy in this case, as the op-tree being operated
    on may be changing at each step. The exact behavior is that it applies the
    rule to the current node, and if nothing changes, it applies it to the
    children. If something did happen, it applies the rule at the current node
    again. This means you *can* accidentally write rules that will loop forever
    if they alternate between multiple equivalent forms each time they are
    applied -- be careful not to do this.
    """

    @wraps(func)
    def wrapped(op: KeySetOp) -> KeySetOp:
        new_op = func(op)
        if new_op != op:
            return wrapped(new_op)

        if isinstance(new_op, (Detect, FromTuples, FromSparkDataFrame)):
            return new_op
        elif isinstance(new_op, CrossJoin):
            return type(new_op)(tuple(wrapped(f) for f in new_op.factors))
        elif isinstance(new_op, Join):
            return Join(wrapped(new_op.left), wrapped(new_op.right))
        elif isinstance(new_op, Project):
            return Project(wrapped(new_op.child), new_op.projected_columns)
        elif isinstance(new_op, Filter):
            return Filter(wrapped(new_op.child), new_op.condition)
        elif isinstance(new_op, Subtract):
            return Subtract(wrapped(new_op.left), wrapped(new_op.right))
        else:
            raise AnalyticsInternalError(
                f"Unhandled KeySetOp subtype {type(new_op).__qualname__} encountered."
            )

    return wrapped


@breadth_first
def project_across_crossjoin(op: KeySetOp) -> KeySetOp:
    """Split projections and move them inside cross-joins.

    If the child of a Project operation is a CrossJoin operation, the Project
    may be split into two smaller Projects over the children of the
    CrossJoin. This significantly improves performance, and allows KeySets like
    ``AB["A"] * C`` and ``(AB * C)["A", "C"]`` to be recognized as equivalent.

    If the Project operation doesn't keep any columns from a CrossJoin factor,
    that factor is dropped. If all but one factor is dropped, the CrossJoin is
    removed altogether.
    """
    if not isinstance(op, Project) or not isinstance(op.child, CrossJoin):
        return op

    included_factors = tuple(
        Project(f, frozenset(f.columns() & op.projected_columns))
        for f in op.child.factors
        if f.columns() & op.projected_columns
    )
    if len(included_factors) == 1:
        return included_factors[0]

    return CrossJoin(included_factors)


@depth_first
def collapse_nested_projections(op: KeySetOp) -> KeySetOp:
    """Combine nested projection operations.

    If the child of a Project operation is another Project operation, the inner
    Project may be ignored because the outer one will end up dropping every
    column that was dropped by the inner one.
    """
    if isinstance(op, Project) and isinstance(op.child, Project):
        return Project(op.child.child, op.projected_columns)
    return op


@depth_first
def remove_noop_projections(op: KeySetOp) -> KeySetOp:
    """Remove projection operations that have no effect.

    If the child of a Project operation has no columns other than the projected
    ones, the Project does nothing and can be removed.
    """
    if isinstance(op, Project) and op.projected_columns == set(op.child.columns()):
        return op.child

    return op


@depth_first
def merge_cross_joins(op: KeySetOp) -> KeySetOp:
    """Merge adjacent CrossJoin operations.

    When one CrossJoin operation is a factor of another, combine the two into a
    single, bigger CrossJoin. Also drops any factors corresponding to total
    aggregations, as these do not affect the resulting set of keys.

    Note that this will convert InMemoryCrossJoin back into CrossJoin if an
    InMemoryCrossJoin is a child of a CrossJoin. This is desirable, as it allows
    recomputing the in-memory groups to potentially include the new factors.
    """
    if not isinstance(op, CrossJoin):
        return op

    factors: list[KeySetOp] = []
    for f in op.factors:
        if isinstance(f, CrossJoin):
            factors.extend(f.factors)
        else:
            factors.append(f)

    non_total_factors = [f for f in factors if len(f.columns()) > 0]

    # If all of the factors correspond to total aggregations, just pick one and
    # return it.
    if len(non_total_factors) == 0:
        return factors[0]
    elif len(non_total_factors) == 1:
        return non_total_factors[0]
    return CrossJoin(tuple(non_total_factors))


@depth_first
def order_cross_joins(op: KeySetOp) -> KeySetOp:
    """Order the factors in each CrossJoin in a standard way."""
    if not isinstance(op, CrossJoin):
        return op

    return type(op)(tuple(sorted(op.factors, key=lambda f: tuple(sorted(f.columns())))))


@depth_first
def apply_cross_joins_in_memory(op: KeySetOp) -> KeySetOp:
    """Apply small cross-joins of FromTuples using InMemoryCrossJoin.

    Assumes that merge_cross_joins has already been applied, as
    total-aggregation factors are not allowed by InMemoryCrossJoin.
    """
    # Don't re-apply this rule to InMemoryCrossJoin, as they're already assumed
    # to be structured appropriately, and don't apply it to plans because
    # optimizing conversion to a dataframe isn't relevant if the KeySetOp can't
    # be converted to a dataframe.
    if (
        not isinstance(op, CrossJoin)
        or isinstance(op, InMemoryCrossJoin)
        or op.is_plan()
    ):
        return op

    large_factors = []
    small_factors = []
    for f in op.factors:
        size = f.size(fast=True)
        if (
            isinstance(f, FromTuples)
            and size is not None
            and size < _IN_MEMORY_CROSS_JOIN_THRESHOLD // 2
        ):
            small_factors.append(f)
        else:
            large_factors.append(f)

    # Not enough factors small enough to optimize, just return the original op
    if len(small_factors) < 2:
        return op

    in_memory_factor_iter = cast(
        list[tuple[int, FromTuples]],
        [(f.size(fast=True), f) for f in small_factors],
    )
    if any(f[0] is None for f in in_memory_factor_iter):
        raise AnalyticsInternalError(
            "Size of CrossJoinFromTuples should always be able to "
            "be determined quickly."
        )

    in_memory_factors = deque(sorted(in_memory_factor_iter, key=lambda t: t[0]))

    while in_memory_factors:
        size, large_f = in_memory_factors.pop()
        group = [large_f]
        while (
            in_memory_factors
            and in_memory_factors[0][0] * size < _IN_MEMORY_CROSS_JOIN_THRESHOLD
        ):
            small_size, small_f = in_memory_factors.popleft()
            size *= small_size
            group.append(small_f)

        if len(group) == 1:
            large_factors.append(group[0])
        else:
            large_factors.append(InMemoryCrossJoin(tuple(group)))

    if len(large_factors) == 1:
        return large_factors[0]
    return CrossJoin(tuple(large_factors))


def normalize_joins(op: KeySetOp) -> KeySetOp:
    r"""Restructure Joins into a consistent layout.

    This rewrite rule applies two related changes to collections of nested
    Join operations to ensure a common structure:
    * The joins are restructured such that only the right subtree of a
      join can be another join.
    * The leaves of this collection of joins are sorted by their columns,
      with the first near the top.

    As an example (where `*` represents a Join operation), it would
    make the following transformation:

    ::

          *                *
         / \              / \
        *   *      ->     A *
       / \ / \             / \
       A C D B             B *
                            / \
                            C D

    Note that this rule also applies to a single non-nested join, which
    would be transformed such that the child whose column list is first in the
    sort becomes the left child and the other becomes the right child.
    """
    if isinstance(op, CrossJoin):
        return type(op)(tuple(normalize_joins(f) for f in op.factors))
    if isinstance(op, Project):
        return Project(normalize_joins(op.child), op.projected_columns)
    if isinstance(op, Filter):
        return Filter(normalize_joins(op.child), op.condition)
    if isinstance(op, (Detect, FromTuples, FromSparkDataFrame)):
        return op
    if isinstance(op, Subtract):
        return Subtract(normalize_joins(op.left), normalize_joins(op.right))

    if not isinstance(op, Join):
        raise AnalyticsInternalError(
            f"Unhandled KeySetOp subtype {type(op).__qualname__} encountered."
        )

    leaves = []
    joins = [op]
    while joins:
        current = joins.pop()
        for child in (current.left, current.right):
            if isinstance(child, Join):
                joins.append(child)
            else:
                leaves.append(normalize_joins(child))

    # Reversing the sort and swapping the right/left parameters in the reduce
    # produces a tree where the topmost leaf is the first in the un-reversed
    # order, for example Join(AB, Join(BC, CD)). There's not a technical reason to
    # prefer that ordering over a different one, it's just the most obvious one
    # when reading off the op-tree.
    leaves.sort(key=lambda v: tuple(sorted(v.columns())), reverse=True)
    return reduce(lambda r, l: Join(l, r), leaves)


_REWRITE_RULES = [
    project_across_crossjoin,
    collapse_nested_projections,
    remove_noop_projections,
    merge_cross_joins,
    order_cross_joins,
    apply_cross_joins_in_memory,
    normalize_joins,
]
"""A list of all rewrite rules that will be applied, in order.

Each element of this list should be a function which takes a KeySetOp and
returns a KeySetOp that represents the same set of keys. The rules are applied
in the order they appear in this list, with the transformed output of each rule
passed as the input to the next rule.
"""


def rewrite(op: KeySetOp) -> KeySetOp:
    """Rewrite the given op-tree into an optimized representation."""
    for r in _REWRITE_RULES:
        op = r(op)
    return op
