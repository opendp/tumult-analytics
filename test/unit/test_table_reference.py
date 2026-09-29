"""Unit tests for table reference helpers."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import List

import pytest
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import (
    AddRemoveKeys,
    DictMetric,
    IfGroupedBy,
    SumOf,
    SymmetricDifference,
)

from tmlt.analytics._table_identifier import NamedTable, TableCollection, TemporaryTable
from tmlt.analytics._table_reference import (
    TableReference,
    find_children,
    find_named_tables,
    find_named_tables_with_domains_and_metrics,
    lookup_domain,
    lookup_metric,
)

_DF_DOMAIN = SparkDataFrameDomain(
    {"id": SparkIntegerColumnDescriptor(), "A": SparkStringColumnDescriptor()}
)
_OTHER_DF_DOMAIN = SparkDataFrameDomain({"B": SparkIntegerColumnDescriptor()})


def _reference_find_named_tables(domain: Domain) -> List[TableReference]:
    """The original implementation of find_named_tables, for comparison."""
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


def _nested_domain_and_metric():
    """A domain and metric resembling those of a Session with ID spaces."""
    temp = TemporaryTable()
    domain = DictDomain(
        {
            NamedTable("rows1"): _DF_DOMAIN,
            TableCollection("space_a"): DictDomain(
                {
                    NamedTable("id_a1"): _DF_DOMAIN,
                    NamedTable("id_a2"): _DF_DOMAIN,
                }
            ),
            NamedTable("grouped"): _OTHER_DF_DOMAIN,
            temp: _DF_DOMAIN,
            TableCollection("space_b"): DictDomain({NamedTable("id_b1"): _DF_DOMAIN}),
            NamedTable("rows2"): _OTHER_DF_DOMAIN,
        }
    )
    metric = DictMetric(
        {
            NamedTable("rows1"): SymmetricDifference(),
            TableCollection("space_a"): AddRemoveKeys(
                {NamedTable("id_a1"): "id", NamedTable("id_a2"): "id"}
            ),
            NamedTable("grouped"): IfGroupedBy(["B"], SumOf(SymmetricDifference())),
            temp: SymmetricDifference(),
            TableCollection("space_b"): AddRemoveKeys({NamedTable("id_b1"): "id"}),
            NamedTable("rows2"): SymmetricDifference(),
        }
    )
    return domain, metric


def test_find_named_tables_matches_reference():
    """find_named_tables returns the same tables, in the same order, as before."""
    domain, _ = _nested_domain_and_metric()
    expected = _reference_find_named_tables(domain)
    assert find_named_tables(domain) == expected
    assert {t.identifier for t in expected} == {
        NamedTable(n) for n in ["rows1", "id_a1", "id_a2", "grouped", "id_b1", "rows2"]
    }


def test_find_named_tables_with_domains_and_metrics():
    """The single walk agrees with per-table lookups from the root."""
    domain, metric = _nested_domain_and_metric()
    expected = [
        (ref, lookup_domain(domain, ref), lookup_metric(metric, ref))
        for ref in _reference_find_named_tables(domain)
    ]
    assert find_named_tables_with_domains_and_metrics(domain, metric) == expected


def test_find_named_tables_with_domains_and_metrics_empty():
    """An empty domain has no named tables."""
    assert not find_named_tables_with_domains_and_metrics(
        DictDomain({}), DictMetric({})
    )


def test_find_named_tables_with_domains_and_metrics_bad_metric():
    """A metric that cannot be referenced into raises an error."""
    domain = DictDomain({NamedTable("t"): _DF_DOMAIN})
    with pytest.raises(ValueError, match="cannot reference into it"):
        find_named_tables_with_domains_and_metrics(domain, SymmetricDifference())
