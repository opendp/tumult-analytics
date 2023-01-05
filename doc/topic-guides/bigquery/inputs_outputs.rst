.. _BigQuery inputs and outputs:

BigQuery inputs and outputs
===========================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

In the second part of this topic guide, we will show how to 
adapt a Tumult Analytics program to use BigQuery inputs and outputs,
and provide a minimal example of a BigQuery-compatible program.

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

First, let us upload a file to BigQuery to then use it in Tumult Analytics.
If you already have a table you want to use, you can skip this section.

1. Go to the `BigQuery interface`_
2. Create a new dataset by clicking on the three dots on the right of your project name and selecting "Create dataset"
3. Name the dataset ``analytics_tutorial``
4. Create a new table by clicking on the three dots on the right of the dataset name and selecting "Create table"
5. In the table creation page, upload the CSV file that you downloaded from https://tumult-public.s3.amazonaws.com/library-members.csv
6. Name the table ``library_members``

We also need to create a Google Cloud Storage bucket to store the 
intermediate results and our programs.

1. Go to the `Cloud Storage interface`_
2. Create a new bucket by clicking on ``+ CREATE``

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
   +  SparkSession
   +  .builder
   +  .config("spark.sql.warehouse.dir", os.path.join("gs://", BUCKET, "/spark-warehouse/"))
   +  .config("temporaryGcsBucket", BUCKET)
   +  .getOrCreate()
   +)

Specifying BigQuery inputs and outputs
--------------------------------------

Then, using BigQuery for inputs/outputs straightforward. Instead of 
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

Parsing remote procedure parameters
-----------------------------------

We will later call our Tumult Analytics program from a remote procedure in 
BigQuery, and pass parameters from BigQuery to our program. To do this, we 
need to read the environmental variables to set the parameters of our program.
Each parameter is stored in the environmental variable in JSON format, and its 
name has the following format: ``BIGQUERY_PROC_PARAM.[PARAMETER NAME]``. For example, 
if we have a parameter named ``epsilon``, we can access it with 
``os.environ["BIGQUERY_PROC_PARAM.epsilon"]``.

In the following snippet, instead of hard-coding the table path, we take in the 
``project``, ``dataset``, and ``table`` as parameters instead.

.. note:: A remote procedure is how you can call our external Spark code from BigQuery. 
    We will explain how to set it up in the third part of this topic guide; to learn more, 
    you can also consult the `BigQuery documentation`_.
.. _BigQuery documentation: https://cloud.google.com/bigquery/docs/spark-procedures#create-spark-procedure

Full example
------------

Here is the full example, with all the necessary changes.

.. The environment variables do not persist
.. testsetup:: [parameters]

  import os

  parameters = {
      "project": "tumult-labs",
      "dataset": "analytics_tutorial",
      "table": "library_members"
  }
  for key, value in parameters.items():
      os.environ[f"BIGQUERY_PROC_PARAM.{key}"] = value

.. code-block:: python

   import json
   import os

   PROJECT = json.loads(os.environ["BIGQUERY_PROC_PARAM.project"])
   DATASET = json.loads(os.environ["BIGQUERY_PROC_PARAM.dataset"])
   TABLE = json.loads(os.environ["BIGQUERY_PROC_PARAM.table"])

..    :hide:
..    :options: -ELLIPSIS, +NORMALIZE_WHITESPACE


With this completed program, we will store it in a Google Cloud Storage bucket 
so that we can call it from BigQuery later.

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

In the :ref:`next part of this topic guide<gcp docker image>`, 
we will see how to create a GCP-compatible Analytics Docker image.