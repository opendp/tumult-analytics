.. _Running the program:

Calling Tumult Analytics from a BigQuery stored procedure
=========================================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

In this section, we will explain how to run a Tumult Analytics 
program directly from BigQuery. We will do so using the sample 
program from the :ref:`second part<bigquery inputs and outputs>` 
of this topic guide.

You should also have data in BigQuery, and a Google Cloud Storage bucket 
to store intermediate results from the previous parts. Let's assume that 
our table is as initialized in the :ref:`BigQuery setup section<bigquery setup>` 
``tumult-labs.analytics_tutorial.library_members``. We'll use the public 
Tumult Labs image being hosted at 
``us-docker.pkg.dev/tumult-labs/analytics/tutorial:demo``.

.. note:: If you want to use your own image, you can find the instructions 
   to do so in the :ref:`Docker section<gcp docker image>` of this topic guide.

.. code-block:: python

    PROJECT          = "tumult-labs"
    BIGQUERY_DATASET = "analytics_tutorial"
    BIGQUERY_TABLE   = "library_members"
    IMAGE_REPOSITORY = "analytics"
    IMAGE_NAME       = "tutorial"

In BigQuery, tables are used to store the data, and datasets are used 
to group tables and procedures together.
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
procedure directly from the BigQuery editor.

For this example, we can ignore the parameters, as our script does not 
take any. With the sample values used throughout this topic guide, and 
choosing ``count_members`` as the name of our stored procedure, we end 
up with the following query.

.. code-block:: sql

    CREATE OR REPLACE PROCEDURE `tumult-labs.analytics_tutorial.count_members`()
    WITH CONNECTION `tumult-labs.us.bigspark`
    OPTIONS (
        engine='SPARK',
        container_image='us-docker.pkg.dev/tumult-labs/analytics/tutorial:demo',
        main_file_uri='gs://tumult-shared-procedures/library_members.py'
    )
    LANGUAGE python

.. note:: When copy-pasting the procedure creation script, make sure you 
  replace the procedure name and external connection name,
  to point to your own project dataset and connection.

This creates a stored procedure that exists in 
``tumult-labs.analytics_tutorial.count_members``, akin to defining a function. 
Finally you can run the remote procedure by calling it with the appropriate parameters.

.. code-block:: sql

    CALL `tumult-labs.analytics_tutorial.count_members`()

If successful, our script should produce a BigQuery table, which we can 
see after a few minutes once we refresh the page. Otherwise, you can 
check `Cloud Logging`_ for the results. This does require you to enable 
the Cloud Logging API as well.

.. _Cloud Logging: https://console.cloud.google.com/logs

Congratulations! You have successfully created a stored procedure 
that runs a Tumult Analytics program in BigQuery. The next few parts 
of the topic guide will cover how to set your :ref:`own parameters<Passing parameters to a stored procedure>` to the 
program passing it from the remote procedure, and creating a custom 
image to include the libraries necessary for your programs.
