"""Query expression for Filter."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any

from pyspark.sql import SparkSession
from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema, analytics_to_spark_schema
from tmlt.analytics.constraints import Constraint

from ._base import (
    QueryExprVisitor,
    SingleChildQueryExpr,
    _validate_child_is_transformation,
)


@dataclass(frozen=True)
class Filter(SingleChildQueryExpr):
    """Returns the subset of the rows that satisfy the condition."""

    condition: str
    """A string of SQL expression specifying the filter to apply to the data.

    For example, the string "A > B" matches rows where column A is greater than
    column B.
    """

    def __post_init__(self):
        """Checks arguments to constructor."""
        _validate_child_is_transformation(self.child)
        check_type(self.condition, str)

    def _validate(self, input_schema: Schema):
        """Validation checks for this QueryExpr."""
        spark = SparkSession.builder.getOrCreate()
        test_df = spark.createDataFrame(
            [], schema=analytics_to_spark_schema(input_schema)
        )
        try:
            test_df.filter(self.condition)
        except Exception as e:
            raise ValueError(f"Invalid filter condition '{self.condition}': {e}") from e

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        input_schema = self.child.schema(catalog)
        self._validate(input_schema)
        return input_schema

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        # Filters can only modify a table by dropping rows, and all current
        # constraints remain valid when rows are dropped, so just return the input
        # constraints unmodified.
        return self.child.constraints(catalog)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_filter(self)
