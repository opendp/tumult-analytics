.. _Working with Sessions:

Working with Sessions
=====================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

This topic guide covers how to work with one of the core abstractions of Tumult
Analytics: :class:`Session <tmlt.analytics.session.Session>`. In particular, we
will demonstrate the different ways that a Session can be initialized and
examined. For a simple end-to-end usage example of a Session, a better place to
start is the :ref:`privacy budget tutorial <Working with privacy budgets>`.

At a high level, a Session allows you to evaluate queries on private data in a
way that satisfies differential privacy. When creating a Session, private data
must first be loaded into it, along with a *privacy budget*. You can then use
pieces of the total privacy budget to evaluate queries and return differentially
private results. Tumult Analytics' privacy promise and its caveats are described
in detail in the :ref:`privacy promise topic guide<Privacy promise>`.

..
    TODO(#1585): Add a link to the topic guide about privacy accounting.


.. testcode::
    :hide:

    # Hidden block for imports to make examples testable.
    import csv
    import os
    import pandas as pd
    import tempfile
    from pyspark.sql import SparkSession
    from tmlt.analytics.privacy_budget import PureDPBudget
    from tmlt.analytics.protected_change import AddOneRow
    from tmlt.analytics.query_builder import ColumnType
    from tmlt.analytics.session import Session

Constructing a Session
----------------------

There are two ways to construct a Session:

* directly by initializing it from a data source
* or using a Session Builder.

Both options are described below -- for even more details, consult the
:class:`Session API Reference <tmlt.analytics.session.Session>`.

Initializing from a data source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sessions are constructed from :class:`Spark DataFrames <pyspark.sql.DataFrame>`.
For example, with a dataframe named :code:`spark_df` you can construct a Session
using :meth:`~tmlt.analytics.session.Session.from_dataframe` as follows:

.. testcode::
    :hide:

    # Hidden block just for testing example code.
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(
        pd.DataFrame(
            [["alice", 20, 4.0],
            ["bob", 30, 3.7],
            ["carol", 40, 3.2]],
            columns=["name", "age", "grade"]
        )
    )

.. testcode::

    session_from_dataframe = Session.from_dataframe(
        privacy_budget=PureDPBudget(2),
        source_id="my_private_data",
        dataframe=spark_df,
        protected_change=AddOneRow(),
    )

When you load a Spark DataFrame into a Session, you don't need to specify the
schema of the source; it is automatically inferred from the DataFrame's schema.
Recall from the :ref:`first tutorial <First steps>` that :code:`source_id` is
simply a unique identifier for the private data that is used when constructing
queries.

Using a Session Builder
^^^^^^^^^^^^^^^^^^^^^^^

For analysis use cases involving only one private data source,
:meth:`~tmlt.analytics.session.Session.from_dataframe` is a convenient way of
initializing a Session. However, when you have multiple sources of data, a
:class:`Session Builder <tmlt.analytics.session.Session.Builder>` may be used
instead. First, create your Builder:

.. testcode::

    session_builder = Session.Builder()

Next, add a private source to it:

.. testcode::

    session_builder = session_builder.with_private_dataframe(
        source_id="my_private_data",
        dataframe=spark_df,
        protected_change=AddOneRow(),
    )

You may add additional private sources to the Session, although this is
a more advanced and uncommon use case. Suppose you had additional private
data stored in a CSV file:

.. code-block::

    name, salary
    alice, 52000
    bob, 75000
    carol, 96000
    ...

.. testcode::
    :hide:

    # Hidden block just for testing example code.
    private_csv_path = os.path.join(tempfile.mkdtemp(), "salary_data.csv")
    with open(private_csv_path, "w", newline='') as f:
        my_csv_writer = csv.writer(f)
        my_csv_writer.writerow(['name','salary'])
        my_csv_writer.writerow(['alice',52000])
        my_csv_writer.writerow(['bob',75000])
        my_csv_writer.writerow(['carol',96000])
        f.flush()

First load the data into a Spark dataframe, then add it to the Session:

.. testcode::

    salary_df = spark.read.csv(private_csv_path, header=True, inferSchema=True)
    session_builder = session_builder.with_private_dataframe(
        source_id="my_other_private_data",
        dataframe=salary_df,
        protected_change=AddOneRow(),
    )

Any data file format supported by Spark can be used with Tumult Analytics this way.
See the Spark `data sources documentation`_ for more details on what formats are supported and the available options for them.

.. _data sources documentation: https://spark.apache.org/docs/latest/sql-data-sources.html

A more common use case is to register public
data with your Session (e.g., for use in join operations with the private source).

.. testcode::
    :hide:

    public_df = spark.createDataFrame(
        pd.DataFrame(
            [["alice", "CA", "USA"],
            ["bob", "NY", "USA"],
            ["carol", "TX", "USA"]],
            columns=["name", "state", "country"]
        )
    )

.. testcode::

    session_builder = session_builder.with_public_dataframe(
        source_id="my_public_data",
        dataframe=public_df,
    )

Public sources can also be added retroactively after a Session is created using
the :meth:`~tmlt.analytics.session.Session.add_public_dataframe` method.

When using a Session Builder, you must specify the overall privacy budget separately:

.. testcode::

    session_builder = session_builder.with_privacy_budget(PureDPBudget(1))

Once your Session is configured, the final step is to build it:

.. testcode::

    session = session_builder.build()


Examining a Session's state
---------------------------

After creation, a Session exposes several pieces of information. You can list the
string identifiers of available private or public data sources using
:meth:`private_sources <tmlt.analytics.session.Session.private_sources>` or
:meth:`public_sources <tmlt.analytics.session.Session.public_sources>`, respectively.

.. testcode::

    print(session.private_sources)
    print(session.public_sources)

.. testoutput::

    ['my_other_private_data', 'my_private_data']
    ['my_public_data']

These IDs will typically be used when constructing queries, to specify which data
source a query refers to. They can also be used to access schema information about
individual data sources, through
:meth:`~tmlt.analytics.session.Session.get_schema`.

.. testcode::

    print(session.get_schema('my_private_data'))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Schema({'name': ColumnDescriptor(column_type=ColumnType.VARCHAR, allow_null=True, allow_nan=False, allow_inf=False),
      'age': ColumnDescriptor(column_type=ColumnType.INTEGER, allow_null=True, allow_nan=False, allow_inf=False),
      'grade': ColumnDescriptor(column_type=ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True)})

As you can see, Schemas contain information about what columns are in the data, what their types are, and whether each column can contain null, NaN, or infinite values.

You can access the underlying DataFrames of public sources directly using
:meth:`public_source_dataframes <tmlt.analytics.session.Session.public_source_dataframes>`.
Note that there is no corresponding accessor for private source DataFrames;
after creating a Session, the private data should *not* be inspected or modified.

The last key piece of information a Session exposes is how much privacy budget
the Session has left. As you evaluate queries, the Session's remaining budget will
decrease. The currently-available privacy budget can be accessed through
:meth:`remaining_privacy_budget <tmlt.analytics.session.Session.remaining_privacy_budget>`.
For example, we can inspect the budget of our Session created from the Builder above:

.. testcode::

    print(session.remaining_privacy_budget)

.. testoutput::

    PureDPBudget(epsilon=1)

We have not evaluated any queries yet using this Session, so the remaining budget
is the same as the total budget that we initialized the Session with earlier.
