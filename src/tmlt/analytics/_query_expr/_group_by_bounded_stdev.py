"""Query expression for GroupByBoundedStdev."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple, Union

from tmlt.core.measurements.aggregations import NoiseMechanism
from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.config import config
from tmlt.analytics.keyset import KeySet

from ._base import (
    GroupedMeasurementQueryExpr,
    QueryExprVisitor,
    _schema_for_groupby,
    _validate_child_is_transformation,
    _validate_groupby,
    _validate_measure_column,
)


class StdevMechanism(Enum):
    """Possible mechanisms for the stdev() aggregation.

    Currently, the
    :meth:`~tmlt.analytics.GroupedQueryBuilder.stdev`
    aggregation uses an additive noise mechanism to achieve differential privacy.
    """

    DEFAULT = auto()
    """The framework automatically selects an appropriate mechanism. This choice
    might change over time as additional optimizations are added to the library.
    """
    LAPLACE = auto()
    """Laplace noise is used."""
    GAUSSIAN = auto()
    """Gaussian noise is used. Not compatible with pure DP."""


@dataclass(frozen=True)
class GroupByBoundedStdev(GroupedMeasurementQueryExpr):
    """Returns bounded stdev of a column for each combination of groupby domains."""

    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to compute the standard deviation over."""
    low: float
    """The lower bound for clamping the ``measure_column``.
    Should be less than ``high``.
    """
    high: float
    """The upper bound for clamping the ``measure_column``.
    Should be greater than ``low``.
    """
    output_column: str = "stdev"
    """The name of the column to store the stdev in."""
    mechanism: StdevMechanism = StdevMechanism.DEFAULT
    """Choice of noise mechanism.

    By DEFAULT, the framework automatically selects an
    appropriate mechanism.
    """
    core_mechanism: Optional[NoiseMechanism] = None
    """The Core mechanism used for this aggregation. Specified during compilation."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        _validate_child_is_transformation(self.child)
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.low, float)
        check_type(self.high, float)
        check_type(self.output_column, str)
        check_type(self.mechanism, StdevMechanism)

        if type(self.low) != type(self.high):
            # If one is int and other is float; silently cast to float
            # We use __setattr__ here to bypass the dataclass being frozen
            object.__setattr__(self, "low", float(self.low))
            object.__setattr__(self, "high", float(self.high))

        if self.low >= self.high:
            raise ValueError(
                f"Lower bound '{self.low}' must be less than "
                f"the upper bound '{self.high}'."
            )

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        _validate_measure_column(self, input_schema)
        return _schema_for_groupby(
            self,
            input_schema,
            {
                self.output_column: ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=False
                )
            },
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_groupby_bounded_stdev(self)
