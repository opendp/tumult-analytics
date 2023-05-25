.. _BigQuery inputs and outputs:

BigQuery inputs and outputs
===========================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

In this section, we will show how to adapt a Tumult Analytics
program to use BigQuery inputs and outputs, and provide a minimal
example of a BigQuery-compatible program.

We will use the simple program from :ref:`the first tutorial<first steps>`
which constructs a differentially private count of the number of
members in a fake dataset containing members of a public library
``library-members``.

.. testcode::

    from pyspark import SparkFiles
    from pyspark.sql import SparkSession
    from tmlt.analytics.privacy_budget import PureDPBudget
    from tmlt.analytics.query_builder import QueryBuilder
    from tmlt.analytics.session import Session

    spark = SparkSession.builder.getOrCreate()

    spark.sparkContext.addFile(
        "https://tumult-public.s3.amazonaws.com/library-members.csv"
    )
    members_df = spark.read.csv(
        SparkFiles.get("library-members.csv"), header=True, inferSchema=True
    )

    session = Session.from_dataframe(
        privacy_budget=PureDPBudget(3),
        source_id="members",
        dataframe=members_df
    )

    count_query = QueryBuilder("members").count()
    total_count = session.evaluate(
        count_query,
        privacy_budget=PureDPBudget(epsilon=1)
    )
    total_count.show()

.. testoutput::
    :hide:
    :options: +NORMALIZE_WHITESPACE

    +-----+
    |count|
    +-----+
    |...|
    +-----+

We will explain what needs to change to adapt this program to work on
BigQuery.

Setup
-----

To be able to write to BigQuery, we need to create a Google Cloud
Storage bucket to store the intermediate results and our programs.

1. Go to the `Cloud Storage interface`_
2. Create a new bucket by clicking on ``+ CREATE``

In this topic guide, we will create two buckets. One to house our
programs, and the other for intermediate materialization. For this
topic guide, we will be calling ours ``tumult-shared-procedures``
and ``tumult-warehouse`` respectively. Since buckets use a global
namespace, you will need to choose a unique name for your bucket.

.. _BigQuery interface: https://console.cloud.google.com/bigquery
.. _Cloud Storage interface: https://console.cloud.google.com/storage

Creating the Spark Session
--------------------------

Our Spark session will use a `Google Cloud Storage`_ bucket to store
intermediate results that are generated and used by Tumult Analytics
to compute the differentially private results. This is done by setting
the ``spark.sql.warehouse.dir`` configuration option.

.. _Google Cloud Storage: https://cloud.google.com/storage

Additionally, writing to BigQuery tables requires an intermediate
buffer to write to, which is also stored in a Google Cloud Storage
bucket. In this case, we can use the same bucket for both purposes.
You will need to replace ``BUCKET`` with your own bucket name.

.. note:: Whenever working with sensitive data, make sure that these
    buckets are securely configured and that unauthorized users
    cannot access them.

.. code-block:: diff

    -spark = SparkSession.builder.getOrCreate()
    +BUCKET = "my-gcs-bucket"
    +spark = (
    +    SparkSession
    +    .builder
    +    .config("spark.sql.warehouse.dir", os.path.join("gs://", BUCKET, "/spark-warehouse/"))
    +    .config("temporaryGcsBucket", BUCKET)
    +    .getOrCreate()
    +)

Specifying BigQuery inputs and outputs
--------------------------------------

Then, using BigQuery for inputs/outputs is straightforward. Instead of
reading from a CSV file, we specify that the format we're reading from is
``BigQuery``, with additional ``option`` properties that we set to indicate
each table path.

Here is a code snippet for reading a BigQuery input.
You will need to replace ``PROJECT``, ``DATASET``, and ``TABLE`` with
your own values.

.. code-block:: diff

    -spark.sparkContext.addFile(
    -    "https://tumult-public.s3.amazonaws.com/library-members.csv"
    -)
    -members_df = spark.read.csv(
    -    SparkFiles.get("library-members.csv"), header=True, inferSchema=True
    -)
    +PROJECT = "tumult-labs"
    +DATASET = "analytics_tutorial"
    +TABLE   = "library_members"
    +members_df = (
    +  spark.read.format("bigquery")
    +  .option("table", f"{PROJECT}:{DATASET}.{TABLE}")
    +  .load()
    +)

And here is a snippet to write to a BigQuery table. Here we write our
counts to ``tumult-labs.analytics_tutorial.library_counts``.

.. code-block:: python

    (
        total_count
        .write.format("bigquery")
        .mode("overwrite")
        .option("table", "tumult-labs:analytics_tutorial.library_counts")
        .save()
    )

The format for table names is ``[PROJECT]:[DATASET].[TABLE]``.

Full example
------------

In the end, your program should look structually similar to this final program.

.. code-block:: python

    import json
    import os

    from pyspark.sql import SparkSession

    from tmlt.analytics.privacy_budget import PureDPBudget
    from tmlt.analytics.query_builder import QueryBuilder
    from tmlt.analytics.session import Session

    BUCKET = "tumult-warehouse"
    INPUT_TABLE = "tumult-labs.analytics_tutorial.library_members"
    OUTPUT_TABLE = "tumult-labs.analytics_tutorial.member_counts"

    spark = (
        SparkSession
        .builder
        .config("spark.sql.warehouse.dir", os.path.join("gs://", BUCKET, "/spark-warehouse/"))
        .config("temporaryGcsBucket", BUCKET)
        .getOrCreate()
    )

    members_df = (
        spark.read.format("bigquery")
        .option("table", INPUT_TABLE)
        .load()
    )

    session = Session.from_dataframe(
        privacy_budget=PureDPBudget(3),
        source_id="members",
        dataframe=members_df
    )

    count_query = QueryBuilder("members").count()
    total_count = session.evaluate(
        count_query,
        privacy_budget=PureDPBudget(epsilon=1)
    )

    (
        total_count
        .write.format("bigquery")
        .mode("overwrite")
        .option("table", OUTPUT_TABLE)
        .save()
    )

In the :ref:`next part of this topic guide<running the program>`,
we will run this script to see it materialize our results in BigQuery.
