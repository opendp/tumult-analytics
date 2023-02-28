"""Defines constraints which result in truncations.

These constraints all in some way limit how many distinct values or repetitions
of the same value may appear in a column, often in relation to some other
column.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from dataclasses import dataclass
from typing import List, Tuple

from typeguard import check_type

from tmlt.analytics._table_identifier import TemporaryTable
from tmlt.analytics._table_reference import TableReference, lookup_metric
from tmlt.analytics._transformation_utils import (
    generate_nested_transformation,
    get_table_from_ref,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import AddRemoveKeys, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.dictionary import (
    AugmentDictTransformation,
    CreateDictFromValue,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    LimitRowsPerGroupValue,
)
from tmlt.core.transformations.spark_transformations.truncation import LimitRowsPerGroup

from ._base import Constraint


def simplify_truncation_constraints(constraints: List[Constraint]) -> List[Constraint]:
    """Remove redundant truncation constraints from a list of constraints."""
    max_rows_per_id, other_constraints = [], []
    for c in constraints:
        if isinstance(c, MaxRowsPerID):
            max_rows_per_id.append(c)
        else:
            other_constraints.append(c)

    if max_rows_per_id:
        other_constraints.append(MaxRowsPerID(min(c.max for c in max_rows_per_id)))
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
        to_symmetric_difference: bool = False,
    ) -> Tuple[Transformation, TableReference]:
        parent_metric = lookup_metric(
            child_transformation.output_metric, child_ref.parent
        )
        if not isinstance(parent_metric, AddRemoveKeys):
            raise ValueError(
                "The MaxRowsPerID constraint can only be applied to tables with "
                "the AddRowsWithID protected change."
            )

        if to_symmetric_difference:
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
