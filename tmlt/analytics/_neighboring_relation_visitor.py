"""Module to define NeighboringRelationVisitors."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Dict, NamedTuple, Union

import sympy as sp
from pyspark.sql import DataFrame
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import AddRemoveKeys as CoreAddRemoveKeys
from tmlt.core.metrics import (
    DictMetric,
    IfGroupedBy,
    Metric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics._neighboring_relation import (
    AddRemoveKeys,
    AddRemoveRows,
    AddRemoveRowsAcrossGroups,
    Conjunction,
    NeighboringRelationVisitor,
)
from tmlt.analytics._table_identifier import Identifier, NamedTable, TableCollection


class _RelationIDVisitor(NeighboringRelationVisitor):
    """Generate identifiers for neighboring relations."""

    def visit_add_remove_rows(self, relation: AddRemoveRows) -> Identifier:
        return NamedTable(relation.table)

    def visit_add_remove_rows_across_groups(
        self, relation: AddRemoveRowsAcrossGroups
    ) -> Identifier:
        return NamedTable(relation.table)

    def visit_str(self, s: str) -> Identifier:  # pylint: disable=no-self-use
        """Visit a string.

        Helper method for relations like AddRemoveKeys that need to get
        identifiers from strings.
        """
        return NamedTable(s)

    def visit_add_remove_keys(self, relation: AddRemoveKeys) -> Identifier:
        return TableCollection(relation.id_space)

    def visit_conjunction(self, relation: Conjunction) -> Identifier:
        # Since conjunctions are automatically flattened, they should never need names.
        raise AssertionError(
            "Conjunctions should never appear as sub-relations. "
            "This is a bug, please let us know so we can fix it!"
        )


class NeighboringRelationCoreVisitor(NeighboringRelationVisitor):
    """A visitor for generating an initial Core state from a neighboring relation."""

    class Output(NamedTuple):
        """A container for the outputs of the visitor."""

        domain: Domain
        metric: Metric
        distance: Any
        data: Any

    def __init__(
        self,
        tables: Dict[str, DataFrame],
        output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    ):
        """Constructor."""
        self.tables = tables
        self.output_measure = output_measure

    def visit_add_remove_rows(self, relation: AddRemoveRows) -> Output:
        """Build Core state from ``AddRemoveRows`` neighboring relation."""
        metric = SymmetricDifference()
        distance = ExactNumber(relation.n)
        data = self.tables[relation.table]
        domain = SparkDataFrameDomain.from_spark_schema(data.schema)
        return self.Output(domain, metric, distance, data)

    def visit_add_remove_rows_across_groups(
        self, relation: AddRemoveRowsAcrossGroups
    ) -> Output:
        """Build Core state from ``AddRemoveRowsAcrossGroups`` neighboring relation."""
        # This is needed because it's currently allowed to pass float-valued
        # stabilities in the per_group parameter (for backwards compatibility).
        # TODO(#2272): Remove this.
        per_group = (
            sp.Rational(relation.per_group)
            if isinstance(relation.per_group, float)
            else relation.per_group
        )
        agg_metric: Union[RootSumOfSquared, SumOf]
        if isinstance(self.output_measure, RhoZCDP):
            agg_metric = RootSumOfSquared(SymmetricDifference())
            distance = ExactNumber(
                per_group * ExactNumber(sp.sqrt(relation.max_groups))
            )
        elif isinstance(self.output_measure, (PureDP, ApproxDP)):
            agg_metric = SumOf(SymmetricDifference())
            distance = ExactNumber(per_group * relation.max_groups)
        else:
            raise TypeError(
                f"The provided output measure {self.output_measure} for this visitor is"
                " not supported."
            )

        metric = IfGroupedBy(relation.grouping_column, agg_metric)
        data = self.tables[relation.table]
        domain = SparkDataFrameDomain.from_spark_schema(data.schema)
        return self.Output(domain, metric, distance, data)

    def visit_add_remove_keys(self, relation: AddRemoveKeys) -> Output:
        """Build Core state from ``AddRemoveKeys`` neighboring relation."""
        distance = ExactNumber(relation.max_keys)
        metric_dict: Dict[Identifier, str] = {}
        data_dict: Dict[Identifier, Any] = {}
        domain_dict: Dict[Identifier, Any] = {}
        for table_name, key_column in relation.table_to_key_column.items():
            table_id = _RelationIDVisitor().visit_str(table_name)
            data = self.tables[table_name]
            domain_dict[table_id] = SparkDataFrameDomain.from_spark_schema(data.schema)
            metric_dict[table_id] = key_column
            data_dict[table_id] = data

        return self.Output(
            DictDomain(domain_dict), CoreAddRemoveKeys(metric_dict), distance, data_dict
        )

    def visit_conjunction(self, relation: Conjunction) -> Output:
        """Build Core state from ``Conjunction`` neighboring relation."""
        domain_dict: Dict[Identifier, Any] = {}
        metric_dict: Dict[Identifier, Any] = {}
        distance_dict: Dict[Identifier, Any] = {}
        data_dict: Dict[Identifier, Any] = {}

        for child in relation.children:
            child_id = child.accept(_RelationIDVisitor())
            child_output = child.accept(self)

            domain_dict[child_id] = child_output.domain
            metric_dict[child_id] = child_output.metric
            distance_dict[child_id] = child_output.distance
            data_dict[child_id] = child_output.data

        return self.Output(
            DictDomain(domain_dict), DictMetric(metric_dict), distance_dict, data_dict
        )
