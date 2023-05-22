"""Defines constraints which result in truncations.

These constraints all in some way limit how many distinct values or repetitions
of the same value may appear in a column, often in relation to some other
column.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import (
    AddRemoveKeys,
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
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    LimitKeysPerGroupValue,
    LimitRowsPerGroupValue,
    LimitRowsPerKeyPerGroupValue,
)
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitKeysPerGroup,
    LimitRowsPerGroup,
    LimitRowsPerKeyPerGroup,
)
from typeguard import check_type

from tmlt.analytics._table_identifier import TemporaryTable
from tmlt.analytics._table_reference import TableReference, lookup_metric
from tmlt.analytics._transformation_utils import (
    generate_nested_transformation,
    get_table_from_ref,
)

from ._base import Constraint


def simplify_truncation_constraints(constraints: List[Constraint]) -> List[Constraint]:
    """Remove redundant truncation constraints from a list of constraints."""
    max_rows_per_id, other_constraints = [], []
    max_groups_per_id: Dict[str, int] = {}
    max_rows_per_group_per_id: Dict[str, int] = {}
    for c in constraints:
        if isinstance(c, MaxRowsPerID):
            max_rows_per_id.append(c)
        elif isinstance(c, MaxGroupsPerID):
            if max_groups_per_id.get(c.grouping_column) is None:
                max_groups_per_id[c.grouping_column] = c.max
            elif max_groups_per_id[c.grouping_column] > c.max:
                max_groups_per_id[c.grouping_column] = c.max
        elif isinstance(c, MaxRowsPerGroupPerID):
            if max_rows_per_group_per_id.get(c.grouping_column) is None:
                max_rows_per_group_per_id[c.grouping_column] = c.max
            elif max_rows_per_group_per_id[c.grouping_column] > c.max:
                max_rows_per_group_per_id[c.grouping_column] = c.max
        else:
            other_constraints.append(c)

    if max_rows_per_id:
        other_constraints.append(MaxRowsPerID(min(c.max for c in max_rows_per_id)))
    if max_groups_per_id:
        for grouping_column, max_groups in max_groups_per_id.items():
            other_constraints.append(MaxGroupsPerID(grouping_column, max_groups))
    if max_rows_per_group_per_id:
        for grouping_column, max_groups in max_rows_per_group_per_id.items():
            other_constraints.append(MaxRowsPerGroupPerID(grouping_column, max_groups))

    return other_constraints


@dataclass(frozen=True)
class MaxRowsPerID(Constraint):
    """A constraint limiting the number of rows associated with each ID in a table.

    This constraint limits how many times each distinct value may appear in the
    ID column of a table with the
    :class:`~tmlt.analytics.protected_change.AddRowsWithID` protected
    change. For example, ``MaxRowsPerID(5)`` guarantees that each ID appears in
    at most five rows. It cannot be applied to tables with other protected changes.
    """

    max: int
    """The maximum number of times each distinct value may appear in the column."""

    def __post_init__(self):
        """Check constructor arguments."""
        check_type("max", self.max, int)
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
        if not isinstance(parent_metric, AddRemoveKeys):
            raise ValueError(
                "The MaxRowsPerID constraint can only be applied to tables with "
                "the AddRowsWithID protected change."
            )

        if update_metric:
            target_table = TemporaryTable()
            transformation = get_table_from_ref(child_transformation, child_ref)
            assert isinstance(transformation.output_domain, SparkDataFrameDomain)
            assert isinstance(transformation.output_metric, IfGroupedBy)
            transformation |= LimitRowsPerGroup(
                transformation.output_domain,
                SymmetricDifference(),
                transformation.output_metric.column,
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

            def gen_tranformation_ark(parent_domain, parent_metric, target):
                return LimitRowsPerGroupValue(
                    parent_domain, parent_metric, child_ref.identifier, target, self.max
                )

            return generate_nested_transformation(
                child_transformation,
                child_ref.parent,
                {AddRemoveKeys: gen_tranformation_ark},
            )


@dataclass(frozen=True)
class MaxGroupsPerID(Constraint):
    """A constraint limiting the number of distinct groups per ID.

    This constraint limits how many times a distinct value may appear in the grouping
    column for each distinct value in the table's ID column. For example,
    ``MaxGroupsPerID("grouping_column", 4)`` guarantees that there are at most four
    distinct values of ``grouping_column`` for each distinct value of ``ID_column``.
    """

    grouping_column: str
    """The name of the grouping column."""
    max: int
    """The maximum number of distinct values in the grouping column
    for each distinct value in the ID column."""

    def __post_init__(self):
        """Check constructor arguments."""
        check_type("grouping_column", self.grouping_column, str)
        check_type("max", self.max, int)
        if self.grouping_column == "":
            raise ValueError("grouping_column cannot be empty")
        if self.max < 1:
            raise ValueError(f"max must be a positive integer, not {self.max}")

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
            if not isinstance(parent_metric, AddRemoveKeys):
                raise ValueError(
                    "The MaxGroupsPerID constraint can only be applied to tables with "
                    "the AddRowsWithID protected change."
                )

            target_table = TemporaryTable()
            transformation = get_table_from_ref(child_transformation, child_ref)
            assert isinstance(transformation.output_domain, SparkDataFrameDomain)
            assert isinstance(transformation.output_metric, IfGroupedBy)
            assert isinstance(
                transformation.output_metric.inner_metric, SymmetricDifference
            )

            inner_metric: Union[SumOf, RootSumOfSquared]
            if use_l2:
                inner_metric = RootSumOfSquared(
                    IfGroupedBy(
                        transformation.output_metric.column, SymmetricDifference()
                    )
                )
            else:
                inner_metric = SumOf(
                    IfGroupedBy(
                        transformation.output_metric.column, SymmetricDifference()
                    )
                )

            transformation |= LimitKeysPerGroup(
                transformation.output_domain,
                IfGroupedBy(self.grouping_column, inner_metric),
                transformation.output_metric.column,
                self.grouping_column,
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

            def gen_tranformation_ark(parent_domain, parent_metric, target):
                return LimitKeysPerGroupValue(
                    parent_domain,
                    parent_metric,
                    child_ref.identifier,
                    target,
                    self.grouping_column,
                    self.max,
                )

            return generate_nested_transformation(
                child_transformation,
                child_ref.parent,
                {AddRemoveKeys: gen_tranformation_ark},
            )


@dataclass(frozen=True)
class MaxRowsPerGroupPerID(Constraint):
    """A constraint limiting rows per unique (ID, grouping column) pair in a table.

    For example, ``MaxRowsPerGroupPerID("group_col", 5)`` guarantees that each
    ID appears in at most five rows for each distinct value in ``group_col``.
    """

    grouping_column: str
    """Name of column defining the groups to truncate."""

    max: int
    """The maximum number of times each distinct value may appear in the column."""

    def __post_init__(self):
        """Check constructor arguments."""
        check_type("max", self.max, int)
        if self.max < 1:
            raise ValueError(f"max must be a positive integer, not {self.max}")
        check_type("grouping_column", self.grouping_column, str)
        if self.grouping_column == "":
            raise ValueError("grouping_column cannot be empty")

    def _enforce(
        self,
        child_transformation: Transformation,
        child_ref: TableReference,
        update_metric: bool = False,
    ) -> Tuple[Transformation, TableReference]:
        if update_metric:
            target_table = TemporaryTable()
            transformation = get_table_from_ref(child_transformation, child_ref)
            assert isinstance(transformation.output_domain, SparkDataFrameDomain)
            assert isinstance(transformation.output_metric, IfGroupedBy)
            assert isinstance(
                transformation.output_metric.inner_metric, (SumOf, RootSumOfSquared)
            )
            assert isinstance(
                transformation.output_metric.inner_metric.inner_metric, IfGroupedBy
            )
            transformation |= LimitRowsPerKeyPerGroup(
                transformation.output_domain,
                transformation.output_metric,
                transformation.output_metric.inner_metric.inner_metric.column,
                self.grouping_column,
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
            if not isinstance(parent_metric, AddRemoveKeys):
                raise ValueError(
                    "The MaxRowsPerGroupPerID constraint can only be applied to tables"
                    " with the AddRowsWithID protected change."
                )

            def gen_tranformation_ark(parent_domain, parent_metric, target):
                return LimitRowsPerKeyPerGroupValue(
                    parent_domain,
                    parent_metric,
                    child_ref.identifier,
                    target,
                    self.grouping_column,
                    self.max,
                )

            return generate_nested_transformation(
                child_transformation,
                child_ref.parent,
                {AddRemoveKeys: gen_tranformation_ark},
            )
