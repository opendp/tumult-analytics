"""Rules for rewriting KeySetOp trees."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from functools import reduce, wraps
from typing import Callable

from tmlt.analytics import AnalyticsInternalError

from ._base import KeySetOp
from ._cross_join import CrossJoin
from ._detect import Detect
from ._filter import Filter
from ._from_dataframe import FromSparkDataFrame
from ._from_tuples import FromTuples
from ._project import Project


def depth_first(f: Callable[[KeySetOp], KeySetOp]) -> Callable[[KeySetOp], KeySetOp]:
    """Recursively apply the given method to an op-tree, depth-first."""

    @wraps(f)
    def wrapped(op: KeySetOp) -> KeySetOp:
        if isinstance(op, (Detect, FromTuples, FromSparkDataFrame)):
            return f(op)
        elif isinstance(op, CrossJoin):
            left = wrapped(op.left)
            right = wrapped(op.right)
            return f(CrossJoin(left, right))
        elif isinstance(op, Project):
            child = wrapped(op.child)
            return f(Project(child, op.projected_columns))
        elif isinstance(op, Filter):
            child = wrapped(op.child)
            return f(Filter(child, op.condition))
        else:
            raise AnalyticsInternalError(
                f"Unhandled KeySetOp subtype {type(op).__qualname__} encountered."
            )

    return wrapped


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


def normalize_cross_joins(op: KeySetOp) -> KeySetOp:
    r"""Restructure CrossJoins into a consistent layout.

    This rewrite rule applies two related changes to collections of nested
    CrossJoin operations to ensure a common structure:
    * The CrossJoins are restructured such that only the right subtree of a
      CrossJoin can be another CrossJoin.
    * The leaves of this collection of CrossJoins are sorted by their columns,
      with the first near the top.

    As an example (where `*` represents a CrossJoin operation), it would make
    the following transformation:

    ::

          *                *
         / \              / \
        *   *    ->       A *
       / \ / \             / \
       A C D B             B *
                            / \
                            C D

    Note that this rule also applies to a single non-nested CrossJoin, which
    would be transformed such that the child whose column list is first in the
    sort becomes the left child and the other becomes the right child.
    """
    if isinstance(op, Project):
        return Project(normalize_cross_joins(op.child), op.projected_columns)
    if isinstance(op, Filter):
        return Filter(normalize_cross_joins(op.child), op.condition)
    if isinstance(op, (Detect, FromTuples, FromSparkDataFrame)):
        return op

    if not isinstance(op, CrossJoin):
        raise AnalyticsInternalError(
            f"Unhandled KeySetOp subtype {type(op).__qualname__} encountered."
        )

    leaves = []
    joins = [op]
    while joins:
        current = joins.pop()
        for child in (current.left, current.right):
            if isinstance(child, CrossJoin):
                joins.append(child)
            else:
                leaves.append(child)

    # Reversing the sort and swapping the right/left parameters in the reduce
    # produces a tree where the topmost leaf is the first in the un-reversed
    # order. For example, for a cross-join with three factors A, B, and C, it
    # produces CrossJoin(A, CrossJoin(B, C)). There's not a technical reason to
    # prefer that ordering over a different one, it's just the most obvious one
    # when reading off the op-tree.
    leaves.sort(key=lambda v: tuple(sorted(v.columns())), reverse=True)
    return reduce(lambda r, l: CrossJoin(l, r), leaves)


_REWRITE_RULES = [
    collapse_nested_projections,
    normalize_cross_joins,
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
