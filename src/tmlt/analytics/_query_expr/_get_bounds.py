"""Query expression for GetBounds."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any, Tuple, Union

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import ColumnDescriptor, Schema
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


@dataclass(frozen=True)
class GetBounds(GroupedMeasurementQueryExpr):
    """Returns approximate upper and lower bounds of a column."""

    groupby_keys: Union[KeySet, Tuple[str, ...]]
    """The keys, or columns list to collect keys from, to be grouped on."""
    measure_column: str
    """The column to get bounds of."""
    lower_bound_column: str
    """The name of the column to store the lower bound in."""
    upper_bound_column: str
    """The name of the column to store the upper bound in."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
        if isinstance(self.groupby_keys, tuple):
            config.features.auto_partition_selection.raise_if_disabled()
        check_type(self.groupby_keys, (KeySet, Tuple[str, ...]))
        check_type(self.measure_column, str)
        check_type(self.lower_bound_column, str)
        check_type(self.upper_bound_column, str)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        _validate_groupby(self, input_schema)
        _validate_measure_column(self, input_schema)
        output_type = input_schema[self.measure_column].column_type
        return _schema_for_groupby(
            self,
            input_schema,
            {
                self.lower_bound_column: ColumnDescriptor(
                    output_type, allow_null=False
                ),
                self.upper_bound_column: ColumnDescriptor(
                    output_type, allow_null=False
                ),
            },
        )

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_get_bounds(self)
