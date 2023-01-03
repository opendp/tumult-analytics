.. _Nulls, NaNs, and infinite values:

Nulls, NaNs, and infinite values
================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

This topic guide details how 
`null values <https://en.wikipedia.org/wiki/Null_(SQL)>`__,
`NaN values <https://en.wikipedia.org/wiki/NaN>`__, and 
`infinite values <https://en.wikipedia.org/wiki/IEEE_754-1985#Positive_and_negative_infinity>`__
(positive and negative infinity) are handled in Tumult Analytics.

.. testcode::
    :hide:

    # Hidden block for imports to make examples testable.
    import pandas as pd
    from pyspark.sql import SparkSession

    from tmlt.analytics.keyset import KeySet
    from tmlt.analytics.privacy_budget import PureDPBudget
    from tmlt.analytics.query_builder import QueryBuilder
    from tmlt.analytics.session import Session

Constructing Sessions with Null, NaN, and infinite values
---------------------------------------------------------

By default, Tumult Analytics allows you to use DataFrames that contain
null values, NaNs, or infinite values. Just pass in a DataFrame that
contains those values:

.. testcode::
    :hide:

    # Hidden block just for testing example code.
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(
        pd.DataFrame(
            [[None, 20, 4.0],
            ["bob", 30, None],
            ["carol", 40, float("inf")]],
            columns=["name", "age", "grade"],
        )
    )

.. testcode::
   
   session_with_nulls = Session.from_dataframe(
       privacy_budget=PureDPBudget(2),
       source_id="my_private_data",
       dataframe=spark_df,
   )
   spark_df.show()

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    +-----+---+--------+
    | name|age|   grade|
    +-----+---+--------+
    | null| 20|     4.0|
    |  bob| 30|     NaN|
    |carol| 40|Infinity|
    +-----+---+--------+

Null, NaN, and infinity values are also allowed in public DataFrames of a session.

Null, NaN, and infinite values in maps and flat maps
----------------------------------------------------

By default, all columns created by
:py:meth:`tmlt.analytics.query_builder.QueryBuilder.map` or
:py:meth:`tmlt.analytics.query_builder.QueryBuilder.flat_map` are assumed to contain
null values. If those columns are DECIMAL columns, they are also
assumed to potentially contain NaN or infinite values.

.. testcode::
    
    query = QueryBuilder("private").map(
        f=lambda row: {"new": row["B"]*1.5},
        new_column_types={"new": "DECIMAL"},
        augment=True,
    ).query_expr
    print(query.schema_new_columns)


.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Schema({'new': ColumnDescriptor(column_type=ColumnType.DECIMAL,
                                    allow_null=True,
                                    allow_nan=True,
                                    allow_inf=True)})

If you pass in a full :py:class:`tmlt.analytics.query_builder.ColumnDescriptor`, then you can specify whether new
columns can contain null, NaN, or infinite values:

.. testcode::

    from tmlt.analytics.query_builder import ColumnDescriptor, ColumnType
    new_column_types = {'new': ColumnDescriptor(
        column_type=ColumnType.DECIMAL,
        allow_null=False,
        allow_nan=False,
        allow_inf=False,
    )}
    query = QueryBuilder("private").map(
        f=lambda row: {"new": row["B"]*1.5},
        new_column_types=new_column_types,
        augment=True,
    ).query_expr
    print(query.schema_new_columns)


.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Schema({'new': ColumnDescriptor(column_type=ColumnType.DECIMAL,
                                    allow_null=False,
                                    allow_nan=False,
                                    allow_inf=False)})

If you do this, it is your responsibility to ensure that the mapping
function does not create null, NaN, or infinite values. Tumult Analytics'
will raise an error if there are null values in columns marked as `allow_null=False`,
NaN values in columns marked as `allow_nan=False`, or infinite values
in columns marked as `allow_inf=False`.

Special case: null values in grouping columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tumult Analytics does not allow you to replace null values in 
:py:meth:`a flat map grouping column <tmlt.analytics.query_builder.QueryBuilder.flat_map>`,
because this could violate Tumult Analytics' stability guarantee.
If your flat map transformation could create null values, you cannot replace
them later.

Null, NaN, and infinite values and aggregations
-----------------------------------------------

Analytics automatically transforms your data when you perform a numerical
aggregation - a sum, variance, average, standard deviation, or quantile -
on columns that contain null, NaN, or infinite values.
This section explains how Analytics handles aggregations when data contains null,
NaN, or infinite values.

:py:meth:`tmlt.analytics.query_builder.QueryBuilder.count` and
:py:meth:`tmlt.analytics.query_builder.QueryBuilder.count_distinct`
do not have special behavior for rows containing nulls, NaNs, or infinite values.
Rows with those values are counted the same as rows without any of those values.

Null and NaN values in aggregations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, all rows that contain a null or NaN value in the `measure_column`
are dropped immediately before aggregation. The following example uses a 
Session with an infinite budget to demonstrate this:

.. testcode::
    :hide:

    # Hidden block for setting up the dataframe
    from pyspark.sql.types import (
        LongType,
        StringType,
        StructField,
        StructType,
    )
    private_data = spark.createDataFrame(
        [["Ambar", "Unknown", None],
        ["Tessa", "Unknown", 3]],
        schema=StructType([
            StructField("name", StringType(), nullable=False),
            StructField("genre", StringType(), nullable=True),
            StructField("checked_out", LongType(), nullable=True),
        ]),
    )

.. testcode::

    session = Session.from_dataframe(
        privacy_budget=PureDPBudget(float("inf")),
        source_id="checkouts",
        dataframe=private_data,
    )
    private_data.show()

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    +-----+-------+-----------+
    | name|  genre|checked_out|
    +-----+-------+-----------+
    |Ambar|Unknown|       null|
    |Tessa|Unknown|          3|
    +-----+-------+-----------+

.. testcode::

    query = QueryBuilder("checkouts").groupby(
        KeySet.from_dict({
            "genre": [
                "Unknown",
            ]
        })
    ).average(
        column="checked_out",
        low=0,
        high=30,
    )
    answer = session.evaluate(query, privacy_budget=PureDPBudget(float("inf")))
    answer.show()

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    +-------+-------------------+
    |  genre|checked_out_average|
    +-------+-------------------+
    |Unknown|                3.0|
    +-------+-------------------+

The row where the genre "Unknown" had ``null`` books checked out has been dropped,
so the average number of Unknown books checked out is 3 - even though originally
there were two rows with Unknown books checked out (one with 3 books
checked out, and one with a null value).

If we instead replace all null values with 0, we get a different result:

.. testcode::

    query = QueryBuilder("checkouts").replace_null_and_nan({
        "checked_out": 0,
    }).groupby(
        KeySet.from_dict({
            "genre": [
                "Unknown",
            ]
        })
    ).average(
        column="checked_out",
        low=0,
        high=30,
    )
    answer = session.evaluate(query, privacy_budget=PureDPBudget(float("inf")))
    answer.show()

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    +-------+-------------------+
    |  genre|checked_out_average|
    +-------+-------------------+
    |Unknown|                1.5|
    +-------+-------------------+

If you want to treat null values as zeroes, you must explicitly replace them
before performing your query.

The same principles apply for NaN values.

Infinite values in aggregations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you perform a sum, variance, average, standard deviation, or quantile
query on data containing infinite values, Analytics clamps those infinite
values to the query's upper and lower bounds. Positive infinity is clamped
to the upper bound, and negative infinity is clamped to the lower bound.
The following example uses a Session with an infinite budget to demonstrate this:

.. testcode::
    :hide:

    # Hidden block for setting up the dataframe
    from pyspark.sql.types import DoubleType
    private_data = spark.createDataFrame(
        [["Ambar", "Science fiction", 5.0],
        ["Tessa", "Science fiction", float("-inf")],
        ["Alfredo", "Science fiction", float("inf")]],
        schema=StructType([
            StructField("name", StringType(), nullable=False),
            StructField("genre", StringType(), nullable=True),
            StructField("checked_out", DoubleType(), nullable=True),
        ]),
    )

.. testcode::

    session = Session.from_dataframe(
        privacy_budget=PureDPBudget(float("inf")),
        source_id="checkouts",
        dataframe=private_data,
    )
    private_data.show()

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    +-------+---------------+-----------+
    |   name|          genre|checked_out|
    +-------+---------------+-----------+
    |  Ambar|Science fiction|        5.0|
    |  Tessa|Science fiction|  -Infinity|
    |Alfredo|Science fiction|   Infinity|
    +-------+---------------+-----------+

.. testcode::

    query = QueryBuilder("checkouts").groupby(
        KeySet.from_dict({
            "genre": [
                "Science fiction",
            ]
        })
    ).sum(
        column="checked_out",
        low=0,
        high=30,
    )
    answer = session.evaluate(query, privacy_budget=PureDPBudget(float("inf")))
    answer.show()

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    +---------------+---------------+
    |          genre|checked_out_sum|
    +---------------+---------------+
    |Science fiction|           35.0|
    +---------------+---------------+

Tessa's ``-Infinity`` books checked out became 0, and Alfredo's ``Infinity``
books checked out became 30, for a total of 35 (5 + 0 + 30).

The example below uses different query bounds:

.. testcode::

    query = QueryBuilder("checkouts").groupby(
        KeySet.from_dict({
            "genre": [
                "Science fiction",
            ]
        })
    ).average(
        column="checked_out",
        low=-15,
        high=10,
    )
    answer = session.evaluate(query, privacy_budget=PureDPBudget(float("inf")))
    answer.show()

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    +---------------+-------------------+
    |          genre|checked_out_average|
    +---------------+-------------------+
    |Science fiction|                0.0|
    +---------------+-------------------+

In this example, Tessa's ``-Infinity`` books checked out becomes -15, and
Alfredo's ``Infinity`` books checked out becomes 10. The average number
of books checked out is therefore 0 (5 + 10 + -15, divided by 3).

If you want infinite values to be treated differently, then you should
explicitly drop infinite values (with
:py:meth:`tmlt.analytics.query_builder.QueryBuilder.drop_infinity`) or
replace them (with
:py:meth:`tmlt.analytics.query_builder.QueryBuilder.replace_infinity`) before
performing your aggregation.
