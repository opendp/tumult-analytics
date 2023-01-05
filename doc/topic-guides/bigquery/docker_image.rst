.. _GCP Docker Image:

Creating a Docker image for GCP
===============================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

In the third part of the topic guide, we will create a Docker 
image to run our Tumult Analytics program on GCP. A working knowledge
of Docker is useful to understand this section, but if you have never
used Docker before, you can still follow the instructions below.

This section is optional, and you can use the default image provided by
Tumult Labs, allowing you to skip this part and proceed to the 
:ref:`next<bigquery stored procedures>`. However, if you want to 
customize the image, or use your own dependencies, you will need 
to create your own image.

.. This image can be customized to your needs, but for this example, we will 
.. stick to the default settings.


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

In the :ref:`next part of this topic guide<bigquery stored procedures>`, 
we will see how to create BigQuery stored procedures.