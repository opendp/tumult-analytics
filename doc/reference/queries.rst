.. _queries:

Building queries
================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2024

The :class:`~tmlt.analytics.QueryBuilder` class allows users to construct
differentially private queries using a PySpark-like syntax.

QueryBuilder implements transformations such as joins, maps, or filters.
Using a transformation method returns a new QueryBuilder with that
transformation applied. To re-use the transformations in a
:class:`~tmlt.analytics.QueryBuilder` as the base for multiple queries, users
can create a view using :meth:`~tmlt.analytics.Session.create_view` and write
queries on that view.

QueryBuilder instances can also have an aggregation like
:meth:`~tmlt.analytics.QueryBuilder.count` applied to them, potentially after a
:meth:`~tmlt.analytics.QueryBuilder.groupby`, yielding an object that can be
passed to :func:`~tmlt.analytics.Session.evaluate` to obtain differentially
private results to the query.

.. currentmodule:: tmlt.analytics

QueryBuilder initialization
---------------------------
:class:`~tmlt.analytics.QueryBuilder` initialization is very simple: the only
argument of the constructor is the table on which to apply the query.

.. autosummary::
   :toctree: api/

   QueryBuilder

Transformations
---------------
QueryBuilders implement a variety of transformations, which all yield a new
QueryBuilder instance with that transformation applied. At this stage, the query
cannot yet be evaluated in a differentially private manner, but users can create
views using :meth:`~tmlt.analytics.Session.create_view` on a transformation.

Schema manipulation
"""""""""""""""""""
Transformations that manipulate table schemas.

.. autosummary::
   :toctree: api/

   QueryBuilder.select
   QueryBuilder.rename

Special value handling
""""""""""""""""""""""
Transformations that replace or remove special column values such as null values, NaN
values, or infinity values.

.. autosummary::
   :toctree: api/

   QueryBuilder.replace_null_and_nan
   QueryBuilder.drop_null_and_nan
   QueryBuilder.replace_infinity
   QueryBuilder.drop_infinity

Filters and maps
""""""""""""""""
Transformations that remove or modify rows of private tables, according to
user-specified predicates or functions.

.. autosummary::
   :toctree: api/

   QueryBuilder.filter
   QueryBuilder.map
   QueryBuilder.flat_map
   QueryBuilder.flat_map_by_id

Binning
"""""""
A transformation that groups together nearby values in numerical, date, or
timestamp columns, according to user-specified bins.

.. autosummary::
   :toctree: api/

   QueryBuilder.bin_column
   BinningSpec

.. _constraints:

Constraints
"""""""""""
:meth:`~tmlt.analytics.QueryBuilder.enforce` truncates the sensitive data to limit the maximum impact
of the protected change. More information about it can be found in the
:ref:`Working with privacy IDs` tutorial.

.. autosummary::
   :toctree: api/

   QueryBuilder.enforce
   Constraint
   MaxRowsPerID
   MaxGroupsPerID
   MaxRowsPerGroupPerID

Joins
"""""
Transformations that join the sensitive data with public, non-sensitive data, or
with another sensitive data source.

.. autosummary::
   :toctree: api/

   QueryBuilder.join_public
   QueryBuilder.join_private

Group-by
--------
A transformation that groups the data by the value of one or more columns. The
group-by keys can be specified using a :class:`KeySet`; more information about
it can be found in the :ref:`Group-by queries` tutorial. The transformation
returns a :class:`GroupedQueryBuilder`, a object representing a partial query on
which only aggregations can be run.

.. autosummary::
   :toctree: api/
   :recursive:

   QueryBuilder.groupby

.. autosummary::
   :toctree: api/
   :recursive:
   :nosignatures:

   KeySet
   GroupedQueryBuilder

.. _aggregations:

Aggregations
------------
These aggregations return a :class:`Query` that can be evaluated with differential
privacy. They can be used after a :meth:`~QueryBuilder.groupby` operation on a
:class:`GroupedQueryBuilder`, or on a :class:`QueryBuilder` directly.

.. autosummary::
   :toctree: api/

   QueryBuilder.count
   QueryBuilder.count_distinct
   QueryBuilder.sum
   QueryBuilder.average
   QueryBuilder.variance
   QueryBuilder.stdev
   QueryBuilder.quantile
   QueryBuilder.median
   QueryBuilder.min
   QueryBuilder.max
   QueryBuilder.histogram
   QueryBuilder.get_groups
   QueryBuilder.get_bounds

Queries and post-processing
---------------------------
These classes are returned by aggregations, and can be passed to
:meth:`~tmlt.analytics.Session.evaluate`. Some of them, notably group-by counts, support
additional post-processing operations that can be performed at the same time as query
evaluation.

.. autosummary::
   :toctree: api/
   :nosignatures:

   Query
   GroupbyCountQuery

.. autosummary::
   :toctree: api/

   GroupbyCountQuery.suppress

.. _evaluating-queries:

Evaluating queries
------------------
The :meth:`~tmlt.analytics.Session.evaluate` method is the main function used to
compute queries with differential privacy. QueryBuilders can also be used to create
views using :class:`~tmlt.analytics.Session.create_view`.

The :class:`~tmlt.analytics.Session` also provides methods to add public tables
and perform parallel partitioning.

.. autosummary::
   :toctree: api/

   Session.evaluate
   Session.create_view
   Session.delete_view
   Session.add_public_dataframe
   Session.partition_and_create
   Session.stop

Column types and descriptors
----------------------------
Objects and classes used to describe the schema of tables in a :class:`Session`.

.. autosummary::
   :toctree: api/
   :nosignatures:

   ColumnType
   ColumnDescriptor
   AnalyticsDefault
