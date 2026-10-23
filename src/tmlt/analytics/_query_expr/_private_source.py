"""Query expression for PrivateSource."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from dataclasses import dataclass
from typing import Any

from typeguard import check_type

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._schema import Schema
from tmlt.analytics.constraints import Constraint, simplify_constraints

from ._base import QueryExpr, QueryExprVisitor


@dataclass(frozen=True)
class PrivateSource(QueryExpr):
    """Loads the private source."""

    source_id: str
    """The ID for the private source to load."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.source_id, str)

        if not self.source_id.isidentifier():
            raise ValueError(
                "The string passed as source_id must be a valid Python identifier: it"
                " can only contain alphanumeric letters (a-z) and (0-9), or underscores"
                " (_), and it cannot start with a number, or contain any spaces."
            )

    def _validate(self, catalog: Catalog):
        """Validation checks for this QueryExpr."""
        if self.source_id in catalog.public_tables:
            raise ValueError(
                f"Table '{self.source_id}' is not a private table, and so cannot be "
                "the starting point of a query"
            )
        if self.source_id not in catalog.private_tables:
            raise ValueError(f"Query references nonexistent table '{self.source_id}'")

    def schema(self, catalog: Catalog) -> Schema:
        """Returns the schema resulting from evaluating this QueryExpr."""
        self._validate(catalog)
        return catalog.private_tables[self.source_id].schema

    def constraints(self, catalog: Catalog) -> frozenset[Constraint]:
        """Returns the constraints on the output of this query."""
        self._validate(catalog)
        return simplify_constraints(catalog.private_tables[self.source_id].constraints)

    def accept(self, visitor: "QueryExprVisitor") -> Any:
        """Visits this QueryExpr with visitor."""
        return visitor.visit_private_source(self)
