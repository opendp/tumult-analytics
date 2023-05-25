.. _BigQuery setup:

Introduction to BigQuery
========================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

BigQuery is a serverless, highly scalable, cost-effective, and fully managed
cloud data warehouse for analytics.

In this first section, we will upload a file to BigQuery to use as input for our DP query.
We will use the same input data as for the Tumult Analytics :ref:`tutorial<first steps>`, which is provided
as a CSV file.

1. Go to the `BigQuery interface`_
2. Create a new dataset by clicking on the three dots on the right of your project name and selecting "Create dataset"
3. Name the dataset ``analytics_tutorial``
4. Create a new table by clicking on the three dots on the right of the dataset name and selecting "Create table"
5. In the table creation page, select "Upload" under "Create table from" and select the CSV file you downloaded from https://tumult-public.s3.amazonaws.com/library-members.csv
6. Name the table ``library_members``
7. Select the file format as "CSV"
8. Under Schema, select "Auto detect"
9. Click on "Create table"

.. _BigQuery interface: https://console.cloud.google.com/bigquery

With our data in place, we can explore the data in BigQuery.
We can expand the dataset we previously created, open up the table
to see the schema, and query the data in the query editor
using SQL.

Now that we've set up our environment in BigQuery, let's move on to the :ref:`next part<bigquery inputs and outputs>`
of the topic guide and see how we would modify a simple Tumult Analtyics program to be able
to run in BigQuery.
