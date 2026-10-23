"""Query expression for SuppressAggregates."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema

from ._base import MeasurementQueryExpr, QueryExpr, QueryExprVisitor
from ._group_by_count import GroupByCount


@dataclass(frozen=True)
class SuppressAggregates(MeasurementQueryExpr):
    """Remove all counts that are less than the threshold."""

    child: GroupByCount
    """The aggregate on which to suppress small counts.

    Currently, only GroupByCount is supported.
    """

    column: str
    """The name of the column to suppress."""

    threshold: float
    """Threshold. All counts less than this will be suppressed."""

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.child, QueryExpr)
        if not isinstance(self.child, GroupByCount):
            raise TypeError(
                "SuppressAggregates is only supported on aggregates that are "
                "GroupByCounts"
            )
        check_type(self.column, str)
        check_type(self.threshold, float)

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        return self.child.schema(catalog)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visit this QueryExpr with visitor."""
        return visitor.visit_suppress_aggregates(self)
