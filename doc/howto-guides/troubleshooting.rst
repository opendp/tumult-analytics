.. _Troubleshooting:

Troubleshooting
===============

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2024

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

Receiving empty dataframes as outputs
-------------------------------------

If you're running Analytics queries and getting empty dataframes as outputs,
this likely indicates that your Spark configuration is incorrect. If you run the installation checker, it should identify this problem.

Receiving empty dataframes is a sign that Spark is writing to an incorrect warehouse directory location.
This is most likely to occur in a setting where multiple machines need to use a shared location as a datastore,
but no external warehouse directory is specified.

The issue can be resolved by providing your desired warehouse directory location when building your Spark session.
For example, to configure the session to use to the S3 location ``s3://my-bucket/spark-warehouse``, you would use the following code:

.. code-block::

        from pyspark.sql import SparkSession

        warehouse_location = "s3://my-bucket/spark-warehouse"

        spark = SparkSession.builder.config(
            "spark.sql.warehouse.dir", warehouse_location
        ).getOrCreate()

This assumes that you have configured Spark with the permissions to interact with the given bucket.

If you are using Hive tables to read and write data, you may instead want to consult
the :ref:`Hive section <hive-tips>` of the :ref:`Spark topic guide <spark>`. For more tips
related to Spark, see the entirety of that guide.


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
