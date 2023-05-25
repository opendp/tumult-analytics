Using Tumult Analytics on BigQuery
==================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

This guide explains how to use Tumult Analytics on BigQuery.
We will guide you to getting a minimal example of a Tumult
Analytics program running on BigQuery, then we will explain how to
modify this program to work with custom parameters and custom
libraries.

Following this topic guide requires your project to have access to
the public preview of the stored procedures for Apache Spark. You
can enroll in the preview by completing the `enrollment form`_.

Throughout this topic guide, you must use the same region for all
the objects we will create and use in Google Cloud Platform (GCP): BigQuery tables,
Cloud Storage buckets, Artifact repositories, etc., must all
reside in the same `GCP region`_.

.. _GCP region: https://cloud.google.com/compute/docs/regions-zones
.. _enrollment form: https://cloud.google.com/bigquery/docs/spark-procedures

Let's get started by setting up the environment in Google Cloud
Platform :ref:`here<gcp setup>`.

.. toctree::
   :maxdepth: 1

   setup
   bigquery-setup
   inputs-outputs
   running-the-program
   parameters
   docker-image
