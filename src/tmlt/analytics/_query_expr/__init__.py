"""Building blocks of the Tumult Analytics query language. Not for direct use.

Defines the :class:`QueryExpr` class, which represents expressions in the Tumult
Analytics query language. QueryExpr and its subclasses should not be directly
constructed; but instead built using a :class:`tmlt.analytics.QueryBuilder`. The
documentation of the :class:`tmlt.analytics.QueryBuilder` provides more information
about the intended semantics of :class:`QueryExpr` objects.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Analytics Contributors 2025-present

from tmlt.analytics._schema import FrozenDict

from ._base import (
    AnalyticsDefault,
    GroupedMeasurementQueryExpr,
    MeasurementQueryExpr,
    QueryExpr,
    QueryExprVisitor,
    Row,
    SingleChildQueryExpr,
)
from ._drop_infinity import DropInfinity
from ._drop_null_and_nan import DropNullAndNan
from ._enforce_constraint import EnforceConstraint
from ._filter import Filter
from ._flat_map import FlatMap
from ._flat_map_by_id import FlatMapByID
from ._get_bounds import GetBounds
from ._get_groups import GetGroups
from ._group_by_bounded_average import AverageMechanism, GroupByBoundedAverage
from ._group_by_bounded_stdev import GroupByBoundedStdev, StdevMechanism
from ._group_by_bounded_sum import GroupByBoundedSum, SumMechanism, sums_as_float
from ._group_by_bounded_variance import GroupByBoundedVariance, VarianceMechanism
from ._group_by_count import CountMechanism, GroupByCount
from ._group_by_count_distinct import CountDistinctMechanism, GroupByCountDistinct
from ._group_by_quantile import GroupByQuantile
from ._join_private import JoinPrivate
from ._join_public import JoinPublic
from ._map import Map
from ._private_source import PrivateSource
from ._rename import Rename
from ._replace_infinity import ReplaceInfinity
from ._replace_null_and_nan import ReplaceNullAndNan
from ._select import Select
from ._suppress_aggregates import SuppressAggregates
