"""Building blocks of the Tumult Analytics query language. Not for direct use.

.. deprecated:: 0.14
    This module will be removed in an upcoming release.
    Import mechanism enums from :mod:`tmlt.analytics.query_builder` instead.
    :class:`QueryExpr` will be removed from the Tumult Analytics public API.

Defines the :class:`QueryExpr` class, which represents expressions in the
Tumult Analytics query language. QueryExpr and its subclasses should not be
directly constructed or deconstructed by most users; interfaces such as
:class:`tmlt.analytics.query_builder.QueryBuilder` to create them and
:class:`tmlt.analytics.session.Session` to consume them provide more
user-friendly features.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import warnings

from tmlt.analytics._query_expr import (
    AnalyticsDefault,
    AverageMechanism,
    CountDistinctMechanism,
    CountMechanism,
    DropInfinity,
    DropNullAndNan,
    EnforceConstraint,
    Filter,
    FlatMap,
    GetBounds,
    GetGroups,
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    JoinPrivate,
    JoinPublic,
    Map,
    PrivateSource,
    QueryExpr,
    QueryExprVisitor,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
    StdevMechanism,
    SumMechanism,
    SuppressAggregates,
    VarianceMechanism,
)

__all__ = [
    "AnalyticsDefault",
    "AverageMechanism",
    "CountDistinctMechanism",
    "CountMechanism",
    "DropInfinity",
    "DropNullAndNan",
    "EnforceConstraint",
    "Filter",
    "FlatMap",
    "GetBounds",
    "GetGroups",
    "GroupByBoundedAverage",
    "GroupByBoundedSTDEV",
    "GroupByBoundedSum",
    "GroupByBoundedVariance",
    "GroupByCount",
    "GroupByCountDistinct",
    "GroupByQuantile",
    "JoinPrivate",
    "JoinPublic",
    "Map",
    "PrivateSource",
    "QueryExpr",
    "QueryExprVisitor",
    "Rename",
    "ReplaceInfinity",
    "ReplaceNullAndNan",
    "Select",
    "StdevMechanism",
    "SumMechanism",
    "SuppressAggregates",
    "VarianceMechanism",
]

warnings.warn(
    "The tmlt.analytics.query_expr module is deprecated. Mechanism enums should "
    "now be imported from tmlt.analytics.query_builder.",
    DeprecationWarning,
    stacklevel=2,
)
