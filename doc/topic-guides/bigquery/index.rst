Using Tumult Analytics on BigQuery
==================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

This topic guide explains how to use Tumult Analytics on BigQuery.
We will outline how to read and write data to BigQuery tables using 
Tumult Analytics, create a Docker image on GCP to run a Tumult 
Analytics program and call this program from BigQuery.

Following this topic guide requires your project to have access to 
the public preview of the stored procedures for Apache Spark. You 
can enroll in the preview by completing the `enrollment form`_.

Throughout this topic guide, you must use the same region for all 
the objects we will create and use in GCP: BigQuery tables, 
Cloud Storage buckets, Artifact repositories, etc., must all 
reside in the same `GCP region`_.

.. _GCP region: https://cloud.google.com/compute/docs/regions-zones
.. _enrollment form: https://cloud.google.com/bigquery/docs/spark-procedures

Let's get started by setting up the environment in Google Cloud 
Platform :ref:`here<bigquery setup>`.

.. toctree::
   :maxdepth: 1

   setup
   inputs_outputs
   docker_image
   stored_procedures