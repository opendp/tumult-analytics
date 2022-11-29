"""Module to define NeighboringRelationVisitors."""
from typing import Any, Dict, Tuple, Union

import sympy as sp
from pyspark.sql import DataFrame

from tmlt.analytics._neighboring_relations import (
    AddRemoveRows,
    AddRemoveRowsAcrossGroups,
    Conjunction,
    NeighboringRelationVisitor,
)
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    DictMetric,
    IfGroupedBy,
    Metric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber


class NeighboringRelationCoreVisitor(NeighboringRelationVisitor):
    """Defines a Neighboring Relation Core Visitor."""

    def __init__(
        self, tables: Dict[str, DataFrame], output_measure: Union[PureDP, RhoZCDP]
    ):
        """Constructor."""
        self.tables = tables
        self.output_measure = output_measure

    def visit_add_remove_rows(
        self, relation: AddRemoveRows
    ) -> Tuple[Domain, Metric, Any, Any]:
        """Returns an input domain, input metric, distance, and input data.

        Relation:
        :class:`~tmlt.analytics._neighboring_relations.AddRemoveRows`
        NeighboringRelation.
        """
        input_metric = SymmetricDifference()
        distance = ExactNumber(relation.n)
        input_table = self.tables[relation.table]
        # convert table data to domain schema
        input_domain = SparkDataFrameDomain.from_spark_schema(input_table.schema)
        return input_domain, input_metric, distance, input_table

    def visit_add_remove_rows_across_groups(
        self, relation: AddRemoveRowsAcrossGroups
    ) -> Tuple[Domain, Metric, Any, Any]:
        """Returns an input domain, input metric, distance, and input data.

        Relation:
        :class:`~tmlt.analytics._neighboring_relations.AddRemoveRowsAcrossGroups`
        NeighboringRelation.
        """
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
        elif isinstance(self.output_measure, PureDP):
            agg_metric = SumOf(SymmetricDifference())
            distance = ExactNumber(per_group * relation.max_groups)
        else:
            raise TypeError(
                f"The provided output measure {self.output_measure} for this visitor is"
                " not supported."
            )

        input_metric = IfGroupedBy(relation.grouping_column, agg_metric)
        input_table = self.tables[relation.table]
        input_domain = SparkDataFrameDomain.from_spark_schema(input_table.schema)
        return input_domain, input_metric, distance, input_table

    def visit_conjunction(
        self, relation: Conjunction
    ) -> Tuple[Domain, Metric, Any, Any]:
        """Returns an input domain, input metric, distance, and input data.

        Relation:
        :class:`~tmlt.analytics._neighboring_relations.Conjunction`
        NeighboringRelation.
        """
        domain_dict: Dict[Union[str, int], Any] = {}
        # map names of children to their child domain
        metric_dict: Dict[Union[str, int], Any] = {}
        # map child names to child input metrics
        distance_dict: Dict[Union[str, int], Any] = {}
        # map child names to child input distances
        input_table_dict: Dict[Union[str, int], Any] = {}
        # map child names to inputs for their respective relations
        # Assign numerical indices here instead of using table names
        # because some indices will be table names, whereas others will be
        # dictionaries mapping table names to DataFrames (e.g. with AddRemoveKeys)
        # in the future.
        for index, child in enumerate(relation.children, 1):
            child_visitor = child.accept(self)
            # if the child returns a SparkDataFrameDomain,
            # we know we can use the table name as index
            if isinstance(child_visitor[0], SparkDataFrameDomain):
                source_id = "".join(
                    child._validate(self.tables)  # pylint: disable=protected-access
                )
                domain_dict[source_id] = child_visitor[0]
                metric_dict[source_id] = child_visitor[1]
                distance_dict[source_id] = child_visitor[2]
                input_table_dict[source_id] = child_visitor[3]
            # otherwise we should use a numerical index
            else:
                domain_dict[index] = child_visitor[0]
                metric_dict[index] = child_visitor[1]
                distance_dict[index] = child_visitor[2]
                input_table_dict[index] = child_visitor[3]
        distance = distance_dict
        input_metric = DictMetric(metric_dict)
        input_table = input_table_dict
        input_domain = DictDomain(domain_dict)

        return input_domain, input_metric, distance, input_table
