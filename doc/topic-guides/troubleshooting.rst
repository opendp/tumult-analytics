.. _Troubleshooting:

Troubleshooting
===============

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

This page lists common issues that can arise when using Tumult Analytics,
and explains how to address them.

Handling large amounts of data
------------------------------

When running Analytics locally on large amounts of data (10 million rows or more),
you might encounter Spark errors like
``java.lang.OutOfMemoryError: GC overhead limit exceeded``
or ``java.lang.OutOfMemoryError: Java heap space``.
It's often possible to successfully run Analytics
locally anyway, by configuring Spark with enough RAM. See our
:ref:`Spark guide <spark>` for more information.


``PicklingError`` on map queries
--------------------------------

Functions used in
:class:`~tmlt.analytics.query_expr.Map` or :class:`~tmlt.analytics.query_expr.FlatMap`
queries cannot reference Spark objects, directly or indirectly. If they do,
you might get errors like this:

    ``_pickle.PicklingError: Could not serialize object: RuntimeError: It appears that you are attempting to reference SparkContext from a broadcast variable, action, or transformation. SparkContext can only be used on the driver, not in code that it run on workers``

or like this:

    ``PicklingError: Could not serialize object: TypeError: can't pickle _thread.RLock objects``

For example, this code will raise an error:

.. code-block::

    from typing import Dict, List
    from pyspark.sql import DataFrame, SparkSession
    from tmlt.analytics.query_builder import ColumnType, QueryBuilder

    class DataReader:

        def __init__(self, filenames: List[str]):
            spark = SparkSession.builder.getOrCreate()
            self.data: Dict[str, DataFrame] = {}
            for f in filenames:
                self.data[f] = spark.read.csv(f)

    reader = DataReader(["a.csv", "b.csv"])
    qb = QueryBuilder("private").map(
        f=lambda row: {"data_files": ",".join(reader.data.keys())},
        new_column_types={"data_files": ColumnType.VARCHAR},
    )
    session.create_view(qb, source_id="my_view", cache=True)

If you re-write the map function so that *no* objects referenced inside the
function have *any* references to Spark objects, the map function will succeed:

.. code-block::

    data_files = ",".join(reader.data.keys())
    qb = QueryBuilder("private").map(
        f=lambda row: {"data_files": data_files},
        new_column_types={"data_files": ColumnType.VARCHAR},
    )
    session.create_view(qb, source_id="my_view", cache=True)

Having problems with something else?
------------------------------------

Ask for help on `our Slack server <https://www.tmlt.dev/slack>`_ in the
**#library-questions** channel!
