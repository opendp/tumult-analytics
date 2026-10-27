"""Defines constraints which result in truncations.

These constraints all in some way limit how many distinct values or repetitions
of the same value may appear in a column, often in relation to some other
column.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023-2025, and the Tumult Analytics Contributors 2025-present

import warnings
from dataclasses import dataclass
from typing import Collection, Dict, FrozenSet, Iterable, Optional, Tuple, Union

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import (
    AddRemoveIDs,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.dictionary import (
    AugmentDictTransformation,
    CreateDictFromValue,
)
from tmlt.core.transformations.spark_transformations.add_remove_ids import (
    LimitGroupsPerIDValue,
    LimitRowsPerGroupPerIDValue,
    LimitRowsPerIDValue,
)
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitGroupsPerID,
    LimitRowsPerGroupPerID,
    LimitRowsPerID,
)
from typeguard import check_type

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._table_identifier import TemporaryTable
from tmlt.analytics._table_reference import TableReference, lookup_metric
from tmlt.analytics._transformation_utils import (
    generate_nested_transformation,
    get_table_from_ref,
)

from ._base import Constraint


def _normalize_grouping_columns(
    grouping_columns: Union[str, Collection[str]],
) -> FrozenSet[str]:
    """Validate a grouping column specification and convert it to a frozenset."""
    check_type(grouping_columns, Union[str, Collection[str]])
    if isinstance(grouping_columns, str):
        columns: Collection[str] = [grouping_columns]
    else:
        columns = grouping_columns
        for c in columns:
            check_type(c, str)
    if len(columns) == 0:
        raise ValueError("grouping_columns cannot be empty")
    if any(c == "" for c in columns):
        raise ValueError("grouping column names cannot be empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError(
            f"grouping_columns cannot contain duplicate columns, got {columns}"
        )
    return frozenset(columns)


def _resolve_grouping_columns(
    constraint_name: str,
    grouping_columns: Optional[Union[str, Collection[str]]],
    grouping_column: Optional[Union[str, Collection[str]]],
) -> FrozenSet[str]:
    """Get the grouping columns from a constructor's arguments.

    ``grouping_column`` is a deprecated alias of ``grouping_columns``; exactly one
    of the two must be given.
    """
    if grouping_columns is not None and grouping_column is not None:
        raise TypeError(
            f"{constraint_name} got both grouping_columns and grouping_column;"
            " only grouping_columns should be given."
        )
    if grouping_column is not None:
        warnings.warn(
            f"The grouping_column argument to {constraint_name} is deprecated, and"
            " will be removed in a future release; use grouping_columns instead.",
            DeprecationWarning,
        )
        grouping_columns = grouping_column
    if grouping_columns is None:
        raise TypeError(
            f"{constraint_name} is missing required argument grouping_columns"
        )
    return _normalize_grouping_columns(grouping_columns)


def _require_max(constraint_name: str, max: Optional[int]) -> int:
    """Check that a constructor's max argument was given and is valid."""
    if max is None:
        raise TypeError(f"{constraint_name} is missing required argument max")
    check_type(max, int)
    if max < 1:
        raise ValueError(f"max must be a positive integer, not {max}")
    return max


def _deprecated_grouping_column(
    constraint_name: str, grouping_columns: FrozenSet[str]
) -> str:
    """Return the single grouping column of a constraint, warning that it's deprecated."""
    warnings.warn(
        f"{constraint_name}.grouping_column is deprecated, and will be removed in a"
        " future release; use grouping_columns instead.",
        DeprecationWarning,
    )
    if len(grouping_columns) != 1:
        raise ValueError(
            f"{constraint_name}.grouping_column cannot be used with multiple grouping"
            f" columns ({', '.join(sorted(grouping_columns))}); use grouping_columns"
            " instead."
        )
    (column,) = grouping_columns
    return column


def _format_grouping_columns(grouping_columns: FrozenSet[str]) -> str:
    """Format grouping columns for display, in a deterministic order."""
    return "frozenset({" + ", ".join(repr(c) for c in sorted(grouping_columns)) + "})"


def simplify_truncation_constraints(
    constraints: Iterable[Constraint],
) -> frozenset[Constraint]:
    """Remove redundant truncation constraints from a set of constraints."""
    max_rows_per_id, other_constraints = set(), set()
    max_groups_per_id: Dict[FrozenSet[str], MaxGroupsPerID] = {}
    max_rows_per_group_per_id: Dict[FrozenSet[str], MaxRowsPerGroupPerID] = {}
    for c in constraints:
        if isinstance(c, MaxRowsPerID):
            max_rows_per_id.add(c)
        elif isinstance(c, MaxGroupsPerID):
            key = c.grouping_columns
            if key not in max_groups_per_id or max_groups_per_id[key].max > c.max:
                max_groups_per_id[key] = c
        elif isinstance(c, MaxRowsPerGroupPerID):
            key = c.grouping_columns
            if (
                key not in max_rows_per_group_per_id
                or max_rows_per_group_per_id[key].max > c.max
            ):
                max_rows_per_group_per_id[key] = c
        else:
            other_constraints.add(c)

    if max_rows_per_id:
        other_constraints.add(MaxRowsPerID(min(c.max for c in max_rows_per_id)))
    other_constraints.update(max_groups_per_id.values())
    other_constraints.update(max_rows_per_group_per_id.values())

    return frozenset(other_constraints)


@dataclass(frozen=True)
class MaxRowsPerID(Constraint):
    """A constraint limiting the number of rows associated with each ID in a table.

    This constraint limits how many times each distinct value may appear in the
    ID column of a table with the
    :class:`~tmlt.analytics.AddRowsWithID` protected
    change. For example, ``MaxRowsPerID(5)`` guarantees that each ID appears in
    at most five rows. It cannot be applied to tables with other protected changes.
    """

    max: int
    """The maximum number of times each distinct value may appear in the column."""

    def __post_init__(self):
        """Check constructor arguments."""
        check_type(self.max, int)
        if self.max < 1:
            raise ValueError(f"max must be a positive integer, not {self.max}")

    def _enforce(
        self,
        child_transformation: Transformation,
        child_ref: TableReference,
        update_metric: bool = False,
    ) -> Tuple[Transformation, TableReference]:
        parent_metric = lookup_metric(
            child_transformation.output_metric, child_ref.parent
        )
        if not isinstance(parent_metric, AddRemoveIDs):
            raise ValueError(
                "The MaxRowsPerID constraint can only be applied to tables with "
                "the AddRowsWithID protected change."
            )

        if update_metric:
            target_table = TemporaryTable()
            transformation = get_table_from_ref(child_transformation, child_ref)
            if not isinstance(transformation.output_domain, SparkDataFrameDomain):
                raise AnalyticsInternalError(
                    "Expected MaxRowsPerID to return a SparkDataFrameDomain, but got "
                    f"{transformation.output_domain} instead."
                )
            if not isinstance(transformation.output_metric, IfGroupedBy):
                raise AnalyticsInternalError(
                    "Expected MaxRowsPerID to return an IfGroupedBy metric, but got "
                    f"{transformation.output_metric} instead."
                )
            transformation |= LimitRowsPerID(
                transformation.output_domain,
                SymmetricDifference(),
                transformation.output_metric.columns,
                self.max,
            )
            transformation = AugmentDictTransformation(
                transformation
                | CreateDictFromValue(
                    transformation.output_domain,
                    transformation.output_metric,
                    key=target_table,
                )
            )
            return transformation, TableReference([target_table])

        else:

            def gen_transformation_ark(parent_domain, parent_metric, target):
                return LimitRowsPerIDValue(
                    parent_domain, parent_metric, child_ref.identifier, target, self.max
                )

            return generate_nested_transformation(
                child_transformation,
                child_ref.parent,
                {AddRemoveIDs: gen_transformation_ark},
            )


@dataclass(frozen=True, init=False)
class MaxGroupsPerID(Constraint):
    """A constraint limiting the number of distinct groups per ID.

    This constraint limits how many distinct values may appear in the grouping
    column(s) for each distinct value in the table's ID column. For example,
    ``MaxGroupsPerID("grouping_column", 4)`` guarantees that there are at most four
    distinct values of ``grouping_column`` for each distinct value of ``ID_column``.

    Groups may be defined by one or more columns. For example,
    ``MaxGroupsPerID(["state", "city"], 4)`` guarantees that there are at most
    four distinct ``(state, city)`` pairs for each distinct value of ``ID_column``.
    """

    grouping_columns: FrozenSet[str]
    """The names of the grouping columns.

    When constructing the constraint, this may be given as a single column name
    or as a list of column names; either way, it is stored as a frozenset."""
    max: int
    """The maximum number of distinct values in the grouping column(s)
    for each distinct value in the ID column."""

    def __init__(
        self,
        grouping_columns: Optional[Union[str, Collection[str]]] = None,
        max: Optional[int] = None,
        *,
        grouping_column: Optional[Union[str, Collection[str]]] = None,
    ):
        """Constructor.

        Args:
            grouping_columns: The name of the grouping column, or a list of names
                if the groups are defined by multiple columns. Required unless
                the deprecated ``grouping_column`` is given instead.
            max: The maximum number of distinct values in the grouping column(s)
                for each distinct value in the ID column. Required.
            grouping_column: Deprecated alias of ``grouping_columns``.

                .. deprecated:: 0.22.0
                    Use ``grouping_columns`` instead.
        """
        object.__setattr__(
            self,
            "grouping_columns",
            _resolve_grouping_columns(
                "MaxGroupsPerID", grouping_columns, grouping_column
            ),
        )
        object.__setattr__(self, "max", _require_max("MaxGroupsPerID", max))

    @property
    def grouping_column(self) -> str:
        """The name of the grouping column.

        .. deprecated:: 0.22.0
            Use :attr:`grouping_columns` instead. This property raises an error if
            the constraint has more than one grouping column.
        """
        return _deprecated_grouping_column("MaxGroupsPerID", self.grouping_columns)

    def __repr__(self) -> str:
        """Return a string representation of this constraint."""
        return (
            "MaxGroupsPerID(grouping_columns="
            f"{_format_grouping_columns(self.grouping_columns)}, max={self.max})"
        )

    def _enforce(
        self,
        child_transformation: Transformation,
        child_ref: TableReference,
        update_metric: bool = False,
        use_l2: bool = False,
    ) -> Tuple[Transformation, TableReference]:
        if update_metric:
            parent_metric = lookup_metric(
                child_transformation.output_metric, child_ref.parent
            )
            if not isinstance(parent_metric, AddRemoveIDs):
                raise ValueError(
                    "The MaxGroupsPerID constraint can only be applied to tables with "
                    "the AddRowsWithID protected change."
                )

            target_table = TemporaryTable()
            transformation = get_table_from_ref(child_transformation, child_ref)
            if not isinstance(transformation.output_domain, SparkDataFrameDomain):
                raise AnalyticsInternalError(
                    "Expected MaxGroupsPerID to return a SparkDataFrameDomain, but got "
                    f"{transformation.output_domain} instead."
                )
            if not isinstance(transformation.output_metric, IfGroupedBy):
                raise AnalyticsInternalError(
                    "Expected MaxGroupsPerID to return an IfGroupedBy metric, but got "
                    f"{transformation.output_metric} instead."
                )
            if not isinstance(
                transformation.output_metric.inner_metric, SymmetricDifference
            ):
                raise AnalyticsInternalError(
                    "Expected MaxGroupsPerID to return an IfGroupedBy with a "
                    "SymmetricDifference inner metric, but got "
                    f"{transformation.output_metric.inner_metric} instead."
                )

            inner_metric: Union[SumOf, RootSumOfSquared]
            if use_l2:
                inner_metric = RootSumOfSquared(
                    IfGroupedBy(
                        transformation.output_metric.columns, SymmetricDifference()
                    )
                )
            else:
                inner_metric = SumOf(
                    IfGroupedBy(
                        transformation.output_metric.columns, SymmetricDifference()
                    )
                )

            transformation |= LimitGroupsPerID(
                transformation.output_domain,
                IfGroupedBy(self.grouping_columns, inner_metric),
                transformation.output_metric.columns,
                self.grouping_columns,
                self.max,
            )
            transformation = AugmentDictTransformation(
                transformation
                | CreateDictFromValue(
                    transformation.output_domain,
                    transformation.output_metric,
                    key=target_table,
                )
            )
            return transformation, TableReference([target_table])

        else:

            def gen_transformation_ark(parent_domain, parent_metric, target):
                return LimitGroupsPerIDValue(
                    parent_domain,
                    parent_metric,
                    child_ref.identifier,
                    target,
                    self.grouping_columns,
                    self.max,
                )

            return generate_nested_transformation(
                child_transformation,
                child_ref.parent,
                {AddRemoveIDs: gen_transformation_ark},
            )


@dataclass(frozen=True, init=False)
class MaxRowsPerGroupPerID(Constraint):
    """A constraint limiting rows per unique (ID, grouping column) pair in a table.

    For example, ``MaxRowsPerGroupPerID("group_col", 5)`` guarantees that each
    ID appears in at most five rows for each distinct value in ``group_col``.

    Groups may also be defined by a combination of several columns, by passing a
    list of column names. For example, ``MaxRowsPerGroupPerID(["state", "city"], 5)``
    guarantees that each ID appears in at most five rows for each distinct
    ``(state, city)`` pair.
    """

    grouping_columns: FrozenSet[str]
    """The names of the columns defining the groups to truncate.

    When constructing the constraint, this may be given as a single column name
    or as a list of column names; either way, it is stored as a frozenset.

    Required unless the deprecated ``grouping_column`` parameter is provided
    instead. Providing both this and ``grouping_column`` is invalid.
    """

    max: int
    """The maximum number of times each distinct value may appear in the column.

    Required.
    """

    def __init__(
        self,
        grouping_columns: Optional[Union[str, Collection[str]]] = None,
        max: Optional[int] = None,
        *,
        grouping_column: Optional[Union[str, Collection[str]]] = None,
    ):
        object.__setattr__(self, "max", _require_max("MaxRowsPerGroupPerID", max))
        object.__setattr__(
            self,
            "grouping_columns",
            _resolve_grouping_columns(
                "MaxRowsPerGroupPerID", grouping_columns, grouping_column
            ),
        )

    @property
    def grouping_column(self) -> str:
        """The name of the grouping column.

        .. deprecated:: 0.22.0
            Use :attr:`grouping_columns` instead. This property raises an error if
            the constraint has more than one grouping column.
        """
        return _deprecated_grouping_column(
            "MaxRowsPerGroupPerID", self.grouping_columns
        )

    def __repr__(self) -> str:
        """Return a string representation of this constraint."""
        return (
            "MaxRowsPerGroupPerID(grouping_columns="
            f"{_format_grouping_columns(self.grouping_columns)}, max={self.max})"
        )

    def _enforce(
        self,
        child_transformation: Transformation,
        child_ref: TableReference,
        update_metric: bool = False,
    ) -> Tuple[Transformation, TableReference]:
        if update_metric:
            target_table = TemporaryTable()
            transformation = get_table_from_ref(child_transformation, child_ref)
            if not isinstance(transformation.output_domain, SparkDataFrameDomain):
                raise AnalyticsInternalError(
                    "Expected MaxRowsPerGroupPerID to return a SparkDataFrameDomain, "
                    f"but got {transformation.output_domain} instead."
                )
            if not isinstance(transformation.output_metric, IfGroupedBy):
                raise AnalyticsInternalError(
                    "Expected MaxRowsPerGroupPerID to return an IfGroupedBy metric, "
                    f"but got {transformation.output_metric} instead."
                )
            if not isinstance(
                transformation.output_metric.inner_metric, (SumOf, RootSumOfSquared)
            ):
                raise AnalyticsInternalError(
                    "Expected MaxRowsPerGroupPerID to return an IfGroupedBy with a "
                    "SumOf or RootSumOfSquared inner metric, but got "
                    f"{transformation.output_metric.inner_metric} instead."
                )
            if not isinstance(
                transformation.output_metric.inner_metric.inner_metric, IfGroupedBy
            ):
                raise AnalyticsInternalError(
                    "Expected MaxRowsPerGroupPerID to return an IfGroupedBy with a "
                    "SumOf or RootSumOfSquared inner metric with an IfGroupedBy inner "
                    "metric, but got a(n) "
                    f"{transformation.output_metric.inner_metric.inner_metric} instead."
                )
            transformation |= LimitRowsPerGroupPerID(
                transformation.output_domain,
                transformation.output_metric,
                transformation.output_metric.inner_metric.inner_metric.columns,
                self.grouping_columns,
                self.max,
            )

            transformation = AugmentDictTransformation(
                transformation
                | CreateDictFromValue(
                    transformation.output_domain,
                    transformation.output_metric,
                    target_table,
                )
            )
            return transformation, TableReference([target_table])

        else:
            parent_metric = lookup_metric(
                child_transformation.output_metric, child_ref.parent
            )
            if not isinstance(parent_metric, AddRemoveIDs):
                raise ValueError(
                    "The MaxRowsPerGroupPerID constraint can only be applied to tables"
                    " with the AddRowsWithID protected change."
                )

            def gen_transformation_ark(parent_domain, parent_metric, target):
                return LimitRowsPerGroupPerIDValue(
                    parent_domain,
                    parent_metric,
                    child_ref.identifier,
                    target,
                    self.grouping_columns,
                    self.max,
                )

            return generate_nested_transformation(
                child_transformation,
                child_ref.parent,
                {AddRemoveIDs: gen_transformation_ark},
            )
