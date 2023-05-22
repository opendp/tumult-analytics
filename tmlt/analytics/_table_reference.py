"""Object to reference Tables."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from dataclasses import dataclass
from typing import List, Optional, Union

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.metrics import (
    AddRemoveKeys,
    DictMetric,
    IfGroupedBy,
    Metric,
    SymmetricDifference,
)

from tmlt.analytics._table_identifier import Identifier, NamedTable


@dataclass(frozen=True)
class TableReference:
    """A way to reference tables via their path."""

    path: List[Identifier]
    """The path for the table reference, provided as a list of Identifiers."""

    @property
    def identifier(self) -> Identifier:
        """Get the identifier, i.e. last path segment, of this reference."""
        if not self.path:
            raise IndexError("Empty TableReference has no identifier")
        return self.path[-1]

    @property
    def parent(self) -> "TableReference":
        """Get the parent of this reference."""
        if not self.path:
            raise IndexError("Empty TableReference has no parent")
        return TableReference(self.path[:-1])

    def __truediv__(
        self, value: Union["TableReference", Identifier]
    ) -> "TableReference":
        """Add additional segment(s) to the path via the / operator."""
        value_path = value.path if isinstance(value, TableReference) else [value]
        return TableReference(self.path + value_path)

    def __str__(self):
        """Returns the string representation of the TableReference."""
        return f"TableReference({' -> '.join(str(p) for p in self.path)})"


def lookup_domain(domain: Domain, ref: TableReference) -> Domain:
    """Return the domain of the target of a reference."""
    path = ref.path
    for i, table in enumerate(path):
        if isinstance(domain, DictDomain):
            domain = domain.key_to_domain[table]
        else:
            raise ValueError(
                f"Domain at {path[:i]} is a {type(domain)}, cannot reference into it."
            )
    return domain


def lookup_metric(metric: Metric, ref: TableReference) -> Metric:
    """Return the metric of the target of a reference."""
    path = ref.path
    for i, table in enumerate(path):
        if isinstance(metric, DictMetric):
            metric = metric.key_to_metric[table]
        elif isinstance(metric, AddRemoveKeys):
            metric = IfGroupedBy(metric.df_to_key_column[table], SymmetricDifference())
        else:
            raise ValueError(
                f"Metric at {path[:i]} is a {type(metric)}, cannot reference into it."
            )
    return metric


def find_children(
    domain: Domain, ref: TableReference
) -> Optional[List[TableReference]]:
    """Return the children of a reference, or None if it references a single table."""
    ref_domain = lookup_domain(domain, ref)
    if isinstance(ref_domain, DictDomain):
        return [ref / table for table in ref_domain.key_to_domain.keys()]
    return None


def find_named_tables(domain: Domain) -> List[TableReference]:
    """Get a list of the names of all named tables in a domain."""
    tables: List[TableReference] = []
    pending = [TableReference([])]
    while pending:
        ref = pending.pop()
        children = find_children(domain, ref)
        if children is None:
            tables.append(ref)
        else:
            pending.extend(children)
    return [t for t in tables if isinstance(t.path[-1], NamedTable)]


def find_reference(table_name: str, domain: Domain) -> Optional[TableReference]:
    """Get a TableReference to the table with the given name."""
    if isinstance(domain, DictDomain):
        for k, d in domain.key_to_domain.items():
            if k == NamedTable(table_name):
                return TableReference([k])
            ref = find_reference(table_name, d)
            if ref is not None:
                return TableReference([k]) / ref
    return None
