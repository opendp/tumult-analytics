.. _GCP setup:

GCP setup
=========

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

When entering Google Cloud Platform (GCP) for the first time,
you will be greeted with your project dashboard.

The search bar on top of the page can be used to search for,
and navigate to, any service in GCP.

In the first part of this topic guide, we will enable the APIs
and permissions that are necessary for Tumult Analytics to work.

First, we will enable all the necessary APIs required for
the rest of the guide. Navigate to the `APIs & Services`_
page, and click on `Enable APIs and Services`. Then,
search for and enable the following APIs.

.. _APIs & Services: https://console.cloud.google.com/apis/dashboard

* Artifact Registry API
* BigQuery API
* BigQuery Connection API
* Cloud Build API
* Cloud Logging API
* Cloud Storage API

Next, we will make sure that we have sufficient permissions to
perform the operations needed for this guide. We can check this
by going to `IAM & Admin`_ then under your account, press the edit
button, shaped as a pencil on the right, and add the following permissions.
If you cannot set the following permissions for yourself,
contact your organization administrator.

.. _IAM & Admin: https://console.cloud.google.com/iam-admin/iam

* Create a connection (BigQuery Connection Admin)
* Read / Write GCS Buckets (Google Storage Admin)
* Read Images (Artifact Registry Reader)
* Run BigQuery jobs (BigQuery Job User)
* Use BigQuery Datasets (BigQuery Data Editor)
* View remote procedures (BigQuery Metadata Viewer)

In the :ref:`next part of this topic guide<bigquery setup>`,
we will see how to use BigQuery and how to upload our dataset to it.
