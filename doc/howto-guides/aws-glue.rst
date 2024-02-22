Using Tumult Analytics on AWS Glue
==================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2024

This guide covers using Tumult Analytics as part of an `AWS Glue <https://aws.amazon.com/glue/>`__ data pipeline.
The instructions use `AWS Glue Studio <https://docs.aws.amazon.com/glue/latest/ug/what-is-glue-studio.html>`__, Glue's visual job editor and management interface, but the general steps apply to other ways of configuring Glue jobs as well.

Setting up requirements
^^^^^^^^^^^^^^^^^^^^^^^

Before creating a Glue job, it may be necessary to set up some additional AWS resources for the pipeline to use.
The job we will build requires two such resources:

* an S3 bucket for storing results, and
* an IAM role that provides permissions necessary for the Glue job to run.

If your organization already has S3 buckets and IAM roles configured for Glue, you can use those, and skip the rest of this section.

First, set up a `new S3 bucket <https://s3.console.aws.amazon.com>`__.
This bucket will be called ``tumult-analytics-glue`` for the rest of this guide, but because S3 bucket names must be unique across all AWS accounts, your bucket should have a different name.

Next, create an `IAM role <https://console.aws.amazon.com/iamv2/home>`__ for Glue:

1. Select "AWS service" as the trusted entity type, and pick the "Glue" use case from the drop-down.
2. On the permissions page, add the "AWSGlueServiceRole" managed policy.
3. Fill in a name (for this guide, we will call it ``tumult-glue``) and description on the review page, and create the role.
4. Finally, open the role and hit "Add permissions -> Create inline policy" to create an additional inline policy that grants read-write access to the previously-created S3 bucket.
   The JSON for an appropriate policy is given below; note that the ``Resource`` field will need to be updated with the name of the bucket you created above:

   .. code-block::

      {
          "Version": "2012-10-17",
          "Statement": [
              {
                  "Sid": "VisualEditor0",
                  "Effect": "Allow",
                  "Action": [
                      "s3:PutObject",
                      "s3:GetObject",
                      "s3:ListBucketMultipartUploads",
                      "s3:AbortMultipartUpload",
                      "s3:GetObjectAttributes",
                      "s3:ListBucket",
                      "s3:DeleteObject",
                      "s3:ListMultipartUploadParts"
                  ],
                  "Resource": [
                      "arn:aws:s3:::tumult-analytics-glue/*",
                      "arn:aws:s3:::tumult-analytics-glue"
                  ]
              }
          ]
      }

   Hit "Next", give the policy a name, and then click "Create policy".


Configuring the Glue job
^^^^^^^^^^^^^^^^^^^^^^^^

When using Tumult Analytics in an AWS Glue job, it is run as a single step in a pipeline, consuming the outputs from existing steps and having its output consumed like any other Glue transform.
In this guide, we will perform the following operations:

* Read two inputs, the library checkouts and books datasets from the tutorials, from S3 data sources.
* Pass these inputs into a Glue custom transform that uses Tumult Analytics to compute differentially private statistics.
* Write the resulting table to an S3 target.

To get started, navigate to the "ETL jobs" page of the `AWS Glue console <https://console.aws.amazon.com/gluestudio/home>`__.
Create a new "Visual with a blank canvas" job.
You should now see a blank canvas which will show the steps of your Glue job once they are created.

..
   TODO(#2458): Update the description of what happens when the warehouse directory is misconfigured.

Before creating job steps, navigate to the "Job details" tab, as some configuration is needed for the job to run.
First, set "IAM Role" to the ``tumult-glue`` role that you created earlier.
Next, ensure that "Glue version" is set to Glue 3.0.
Here you can also configure the size and number of workers that will be used; for this guide the smallest worker type and two workers is enough.
Last and most importantly, open the "Advanced properties" section and add two job parameters:

* The ``--additional-python-modules`` key, with value ``tmlt.analytics``.
  This tells Glue to install Tumult Analytics before running the job.
  Version specifiers can be included in this value, for example ``tmlt.analytics==0.7.2``.

* The ``--conf`` key, which sets Spark configuration values, with the value ``spark.sql.warehouse.dir=s3://tumult-analytics-glue/glue-warehouse/``.
  Again, remember to update the bucket name to the one you created earlier.
  This configures Spark to use a warehouse directory that all of the workers can access, which Tumult Analytics needs for storing the results of intermediate computations.
  *If this is not set correctly, evaluating queries may produce empty dataframes.*

Once all of the job settings are updated, return to the "Visual" tab and add the following nodes:

* An `"Amazon S3" data source <https://docs.aws.amazon.com/glue/latest/ug/edit-jobs-source-s3-files.html>`__ named "checkouts" with S3 URL ``s3://tumult-public/checkout-logs.csv`` and CSV data format.

* An "Amazon S3" data source named "books" with S3 URL ``s3://tumult-public/library_books.csv`` and CSV data format.

* A `"Custom Transform" node <https://docs.aws.amazon.com/glue/latest/ug/transforms-custom.html>`__ which has both of the data source nodes as parents.
  This will contain our Tumult Analytics program, which will be added shortly.

* A `"Select from Collection" node <https://docs.aws.amazon.com/glue/latest/ug/transforms-configure-select-collection.html>`__ whose parent is the custom transform node.
  This handles converting the custom transform output, which is required to be a collection of tables, back into a single table for further processing.

* An `"Amazon S3" data target <https://docs.aws.amazon.com/glue/latest/ug/data-target-nodes.html>`__ whose parent is the select from collection node, with the desired data format and S3 target location.
  Here, the target location might be ``s3://tumult-analytics-glue/output/``, but it can be anywhere in S3 that your Glue IAM role can write to.

Once all of this is done, the job should look something like this:

.. image:: ../images/glue_graph.png
   :scale: 70%
   :alt: A view of the Glue Studio console, showing a job with two "Data source - S3 bucket" input nodes, one labeled "books" and the other "checkouts".
         Below these is a "Transform - Custom code" node labeled "dp-analytics" with arrows pointing to it from the two input nodes.
         Below this is a "Transform - SelectFromCollection" node labeled "select-dp-output" with an arrow from the custom code node pointing to it.
         Last, below this is a "Data target - S3 bucket" node labeled "output" with an arrow from the SelectFromCollection pointing to it.
   :align: center

|

With all of the job steps in place, we just need to make the custom transform use Tumult Analytics to compute the desired statistics.
The code to do this looks very much like a typical Tumult Analytics program, but it has some extra steps at the start and end to integrate with other Glue nodes.

.. code-block::

   # Get Spark DataFrames from the DynamicFrameCollection passed into this node:
   checkouts_key = next(k for k in dfc.keys() if k.startswith("checkouts_node"))
   books_key = next(k for k in dfc.keys() if k.startswith("books_node"))
   checkouts_df = dfc[checkouts_key].toDF()
   books_df = dfc[books_key].toDF()

   # Import from Tumult Analytics as usual:
   from tmlt.analytics.session import Session
   from tmlt.analytics.protected_change import AddRowsWithID
   from tmlt.analytics.constraints import MaxRowsPerID
   from tmlt.analytics.query_builder import QueryBuilder
   from tmlt.analytics.privacy_budget import PureDPBudget
   from tmlt.analytics.keyset import KeySet

   # Configure a Session and KeySet based on DataFrames created above:
   budget = PureDPBudget(2)
   session = Session.from_dataframe(
       budget, "checkouts", checkouts_df, protected_change=AddRowsWithID("member_id")
   )
   keyset = KeySet.from_dataframe(books_df.select("title", "author", "isbn"))

   # Evaluate a query:
   output = session.evaluate(
       QueryBuilder("checkouts").enforce(MaxRowsPerID(20)).groupby(keyset).count(),
       budget,
   )

   # Wrap the query output in a DynamicFrameCollection and return it to Glue:
   output_dyn = DynamicFrame.fromDF(output, glueContext, "dp-results")
   return DynamicFrameCollection({"dp-results": output_dyn}, glueContext)

The first block in this script gets DataFrames from the sources connected as parent nodes in Glue.
The complexity comes from the fact that ``DynamicFrameCollection`` keys are not simply the names of the nodes they come from, but rather are formatted ``<node-name>_node<id>`` (for example, ``books_node1685998742192``).
Here, ``id`` is a number assigned to each node, which is not known ahead-of-time and can change if a node is recreated.
For this reason, it's easier to find the right keys by prefix than to look up their names manually.

Once the DataFrames are loaded, the imports, Session initialization, and query evaluation steps all work as in any other Tumult Analytics program.
The query output is then converted into a ``DynamicFrame``, which is placed in a ``DynamicFrameCollection`` and returned to Glue.
The name used (here, ``dp-results``) is not important.

Running the job
^^^^^^^^^^^^^^^

At this point, everything should be in place to run the job you have created.
Give it a name if you haven't already, click "Save", then "Run".
Move over to the "Runs" tab and select the run you just started.
It will take a couple of minutes to start up, but should then succeed and write its output into the configured location in S3.

As you can see, Tumult Analytics provides a convenient way to integrate differential privacy into your AWS Glue ETL jobs.
While the job shown here is relatively simple, you can also use Tumult Analytics as part of larger and more complex jobs that involve many inputs and outputs and use other Glue features for post-processing or other transformations.

If you encounter any unexpected issues, please let us know by reaching out on our `Slack server <https://tmlt.dev/slack>`__ or `filing an issue <https://gitlab.com/tumult-labs/analytics/-/issues>`__.
