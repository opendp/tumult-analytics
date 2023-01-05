.. _BigQuery stored procedures:

Calling Tumult Analytics from a BigQuery stored procedure
=========================================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

In the fourth and final part of this topic guide, we will explain how to run a 
Tumult Analytics program directly from BigQuery. We will do so using the 
sample program from the :ref:`second part<bigquery inputs and outputs>` 
of this topic guide, and the Docker image built in the 
:ref:`third part<gcp docker image>`.

You should also have data in BigQuery, and a Google Cloud Storage bucket 
to store intermediate results from the previous parts. Let's assume that 
our table is ``tumult-labs.analytics_tutorial.library_members``, 
our repository name is ``analytics``, and image name is ``tutorial``.

.. code-block:: python

    PROJECT = "tumult-labs"
    DATASET = "analytics_tutorial"
    TABLE = "library_members"
    REPOSITORY_NAME = "analytics"
    IMAGE_NAME = "tutorial"

In BigQuery, tables are used to store the data, and datasets are used 
to group tables together.
To call external Spark-based programs from BigQuery, we must create 
a *stored procedure*, which is associated with a BigQuery dataset.

First, we need to construct an external data source pointing to Apache Spark.

1. Press the "+ Add Data" button in the top left corner of the `BigQuery console`_.
2. Choose "Connections to external data sources".
3. Select `Apache Spark` as the connection type.
4. Choose a name for the connection, and remember it.
   In our running example, we will call it ``bigspark``.
5. Create the connection.

.. _BigQuery console: https://console.cloud.google.com/bigquery

After creating the connection, in the explorer to the left above our dataset, 
there is now an "External connections" section, in which we can see our 
Apache Spark connection. Its name is the connection name appended 
with the region. In our example, it is ``us.bigspark``, as our connection name is 
``bigspark`` and it is situated in the ``us`` region.

Another thing we need to to with the connection is to copy the service account ID
that was generated for this connection. We will need to grant this service account
the necessary permissions it needs to run our Tumult Analytics program.

To do so, we have to go to the `IAM & Admin`_ page, click "Grant access", paste 
our service account ID in "New Principals", and assign it the following roles.

.. _IAM & Admin: https://console.cloud.google.com/iam-admin/iam

* BigQuery Data Editor
* BigQuery Read Session User
* BigQuery Job User
* Storage Admin
* Artifact Registry Reader

Now, we can navigate back to the BigQuery page to create the stored 
procedure directly from the BigQuery editor. The general syntax to 
do so is as follows.

* ``[PROJECT NAME]`` is the name of your Google Cloud project.
* ``[DATASET NAME]`` is the name of the dataset you want to create the stored 
  procedure in.
* ``[PROCEDURE NAME]`` is the name of the stored procedure.
* ``[PARAMETERS]`` is a comma separated list of parameters for the stored 
  procedure. See this `documentation`_ for more details on valid parameter 
  types.
* ``[CONNECTOR REGION]`` is the region of the connector you created in the 
  previous step.
* ``[CONNECTOR NAME]`` is the name of the connector you created in the previous 
  step.
* ``[IMAGE PATH]`` is the path to the Docker image you created in the previous 
  step. In case you want to use our publicly available image instead, you can 
  use ``us-docker.pkg.dev/tumult-labs/analytics/tutorial:latest``.
* ``[SCRIPT PATH]`` is the path to the script you want to run. This should be 
  the path to the script stored in the Google Cloud Storage bucket in 
  :ref:`the first part<bigquery inputs and outputs>` of this topic guide.

.. _documentation: https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#json_encodings

.. code-block:: sql

    CREATE OR REPLACE PROCEDURE
    	`[PROJECT NAME].[DATASET NAME].[PROCEDURE NAME]`([PARAMETERS])
    WITH CONNECTION `[PROJECT NAME].[CONNECTOR REGION].[CONNECTOR NAME]` OPTIONS (engine='SPARK',
    		container_image='us-docker.pkg.dev/tumult-labs/analytics/tutorial:demo',
    		main_file_uri='[SCRIPT PATH]')
    	LANGUAGE python

With the sample values used throughout this topic guide, and choosing 
``count_members`` as the name of our stored procedure, we end up with 
the following query.

.. code-block:: sql

    CREATE OR REPLACE PROCEDURE
      `tumult-labs.analytics_tutorial.count_members`(
        bucket STRING,
        project STRING,
        dataset STRING,
        table STRING)
    WITH CONNECTION `tumult-labs.us.bigspark` OPTIONS (
        engine='SPARK',
        container_image='us-docker.pkg.dev/tumult-labs/analytics/tutorial:demo',
        main_file_uri='gs://tumult-labs/analytics/library_members.py')
    LANGUAGE python

This creates a stored procedure that exists in 
``[PROJECT NAME].[DATASET NAME].[FUNCTION NAME]``, akin to defining a function. 
Later you will be "calling" this function.

Finally you can run the function by calling it with the appropriate parameters.

.. code-block:: sql

    CALL `[PROJECT NAME].[DATASET NAME].[FUNCTION NAME]`([PARAMETERS])

In our prior example, we would call the function like this:

.. code-block:: sql

    CALL `tumult-labs.analytics_tutorial.count_members`(
        "bucket",
        "tumult-labs",
        "analytics_tutorial",
        "library_members"
    )

If successful, our script should produce a BigQuery table, which we can 
see after a few minutes once we refresh the page. Otherwise, you can 
check `Cloud Logging`_ for the results. This does require you to enable 
the Cloud Logging API as well.

.. _Cloud Logging: https://console.cloud.google.com/logs