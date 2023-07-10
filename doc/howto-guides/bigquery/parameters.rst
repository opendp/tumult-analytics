.. _Passing parameters to a stored procedure:

Passing parameters to a stored procedure
========================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

In this additional section of the topic guide, we will explain how
to pass parameters from BigQuery to the stored procedure containing
our Tumult Analytics program. This makes it possible to customize the
call, by specifying e.g. different inputs or outputs table, or privacy
parameters, without modifying the underlying program.

Recall that our remote procedure from :ref:`earlier<running the program>` had no parameters.

.. code-block:: sql

    CREATE OR REPLACE PROCEDURE `tumult-labs.analytics_tutorial.count_members`()
    WITH CONNECTION `tumult-labs.us.bigspark`
    OPTIONS (
        engine='SPARK',
        container_image='us-docker.pkg.dev/tumult-labs/analytics/tutorial:demo',
        main_file_uri='gs://tumult-shared-procedures/library_members.py'
    )
    LANGUAGE python

We want to add three parameters, being the ``bucket`` where the Spark
warehouse is located, the ``input`` where the input data is located, and the
``output`` where the output data is located.

To do this, we simply need to add the parameters to the procedure definition.

.. code-block:: sql

    CREATE OR REPLACE PROCEDURE `tumult-labs.analytics_tutorial.count_members`(
        bucket STRING,
        input STRING,
        output STRING
    )
    WITH CONNECTION `tumult-labs.us.bigspark`
    OPTIONS (
        engine='SPARK',
        container_image='us-docker.pkg.dev/tumult-labs/analytics/tutorial:demo',
        main_file_uri='gs://tumult-shared-procedures/library_members.py'
    )
    LANGUAGE python

Now, we can call the procedure with the parameters as follows.

.. code-block:: sql

    CALL `tumult-labs.analytics_tutorial.count_members`(
        "tumult-warehouse",
        "tumult-labs.analytics_tutorial.library_members",
        "tumult-labs.analytics_tutorial.member_counts"
    )

.. note:: Replace the bucket, input, and output with the values
    specific to your project.

Now, recall our Tumult Analytics program defined :ref:`earlier<bigquery inputs and outputs>`.

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

We need to modify this so that we can receive the parameters ``bucket``, ``input``, and ``output``.
To read in our new parameters, we need to read the environment variables.
Each parameter is stored in the environment variable in JSON format, and its
name has the following format: ``BIGQUERY_PROC_PARAM.[PARAMETER NAME]``. For example,
if we have a parameter named ``epsilon``, we can access it with
``os.environ["BIGQUERY_PROC_PARAM.epsilon"]``.

.. code-block:: diff

    +import json
    +import os

    -BUCKET = "tumult-warehouse"
    -INPUT_TABLE = "tumult-labs.analytics_tutorial.library_members"
    -OUTPUT_TABLE = "tumult-labs.analytics_tutorial.member_counts"
    +BUCKET = json.loads(os.environ["BIGQUERY_PROC_PARAM.bucket"])
    +INPUT_TABLE = json.loads(os.environ["BIGQUERY_PROC_PARAM.input"])
    +OUTPUT_TABLE = json.loads(os.environ["BIGQUERY_PROC_PARAM.output"])

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

   BUCKET = json.loads(os.environ["BIGQUERY_PROC_PARAM.bucket"])
   INPUT_TABLE = json.loads(os.environ["BIGQUERY_PROC_PARAM.input"])
   OUTPUT_TABLE = json.loads(os.environ["BIGQUERY_PROC_PARAM.output"])

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

In the :ref:`final part of this topic guide<gcp docker image>`,
we will see how to create a customized GCP-compatible Docker image
to run Tumult Analytics.
