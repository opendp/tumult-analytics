"""Query expression for GroupByCountDistinct."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Tuple, Union

from tmlt.core.measurements.aggregations import NoiseMechanism
from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.config import config
from tmlt.analytics.keyset import KeySet

from ._base import (
    AutoColumnQueryExpr,
    GroupedMeasurementQueryExpr,
    QueryExprVisitor,
    schema_for_groupby,
    validate_child_is_transformation,
    validate_groupby,
)


class CountDistinctMechanism(Enum):
    """Enumerating the possible mechanisms used for the count_distinct aggregation.

    Currently, the
    :meth:`~tmlt.analytics.GroupedQueryBuilder.count_distinct`
    aggregation uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Double-sided geometric noise is used."""
    GAUSSIAN = auto()
    """The discrete Gaussian mechanism is used. Not compatible with pure DP."""


@dataclass(frozen=True)
class _CountDistinctBase(GroupedMeasurementQueryExpr):
    """Shared arguments for distinct counts."""

    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    output_column: str = field(default="count_distinct", kw_only=True)
    """The name of the column to store the distinct counts in."""
    mechanism: CountDistinctMechanism = field(
        default=CountDistinctMechanism.DEFAULT, kw_only=True
    )
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an appropriate mechanism.
    """
    core_mechanism: Optional[NoiseMechanism] = field(default=None, kw_only=True)
    """The Core mechanism used for this aggregation. Specified during compilation."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        validate_child_is_transformation(self.child)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.output_column, str)
        check_type(self.mechanism, CountDistinctMechanism)


@dataclass(frozen=True)
class GroupByCountDistinct(_CountDistinctBase):
    """Returns the count of distinct rows in each groupby domain value."""

    columns_to_count: Tuple[str, ...]
    """The columns compared when determining if two rows are distinct."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        super().__post_init__()
        check_type(self.columns_to_count, Tuple[str, ...])

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        validate_groupby(self, input_schema)
        return schema_for_groupby(
            self,
            input_schema,
            {
                self.output_column: ColumnDescriptor(
                    ColumnType.INTEGER, allow_null=False
                )
            },
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_count_distinct(self)


@dataclass(frozen=True)
class AutoGroupByCountDistinct(AutoColumnQueryExpr, _CountDistinctBase):
    """Count distinct rows by comparing all input columns."""

    def resolve_columns(self, catalog: Catalog) -> GroupByCountDistinct:
        """Select all input columns for the distinct count."""
        return GroupByCountDistinct(
            self.child,
            self.groupby_keys,
            tuple(self.child.schema(catalog)),
            output_column=self.output_column,
            mechanism=self.mechanism,
            core_mechanism=self.core_mechanism,
        )
