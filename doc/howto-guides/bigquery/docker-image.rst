.. _GCP Docker Image:

Creating a Docker image for GCP
===============================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

In this final part of the topic guide, we will customize our own
Docker image to contain additional libraries and dependencies.
A working knowledge of Docker is useful to understand this section,
but if you have never used Docker before, you can still follow the
instructions below.

The instructions below show how the public Tumult Analytics demo
image was created, and can be used as a template for your own image.

First, we will need to create a new Docker repository in the `Artifact Registry`_.

.. _Cloud Build: https://console.cloud.google.com/cloud-build
.. _Artifact Registry: https://console.cloud.google.com/artifacts

Next, we will create the image that will be placed in the repository. We will need
to create two files for this locally. The first, ``Dockerfile``, contains Docker
instructions to build the image.

.. code-block:: dockerfile

    FROM python:3.9-bullseye

    # Install the dependencies needed for GCP
    RUN apt-get update && apt-get install -y procps tini

    # Install Tumult Analytics
    RUN pip install --upgrade pip && \
        pip install tmlt.analytics

    # Add additional dependancies here as needed
    # RUN pip install <package>

    # Set up the Spark user for GCP integration
    RUN useradd -ms /bin/bash spark -u 1099
    USER 1099
    WORKDIR /home/spark
    ENV PYSPARK_PYTHON="/usr/local/bin/python"

The second file, ``cloudbuild.yaml``, contains the instructions for the
Google command line tool to build the image and place it in the repository.
In our example, we named our repository ``analytics``, and our image ``tutorial``.
You will need to replace ``REPOSITORY NAME`` and ``IMAGE NAME`` with your
repository and image names you set earlier. We do not need to set the
``$PROJECT_ID`` variable as it is automatically set by the Google command line.

.. code-block:: yaml

    steps:
    - name: 'gcr.io/cloud-builders/docker'
      args: ['build', '-t', 'us-docker.pkg.dev/$PROJECT_ID/[REPOSITORY NAME]/[IMAGE NAME]', '.']
    images:
    - 'us-docker.pkg.dev/$PROJECT_ID/[REPOSITORY NAME]/[IMAGE NAME]'

Then, to build the image, we need to install the `Google Cloud CLI tool`_.

.. _Google Cloud CLI tool: https://cloud.google.com/sdk/docs/install-sdk

Finally, we can build the image by running the following command.

.. code-block:: bash

    gcloud builds submit --region=global --config cloudbuild.yaml --project=[PROJECT NAME]

If it ran successfully, you should see a new completed image in the `Artifact Registry`_.

.. _Artifact Registry: https://console.cloud.google.com/artifacts

With the guide complete, you should be able to run Tumult Analytics
programs on Google Cloud Platform. If you have any questions,
please feel free to join our `Slack`_, or contact us at `our support email`_.

.. _Slack: https://tmlt.dev/slack
.. _our support email: support@tmlt.io
