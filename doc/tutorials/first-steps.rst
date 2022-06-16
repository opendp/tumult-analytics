.. _First steps:

Tutorial 1: First steps with Tumult Analytics
=============================================

In this first tutorial, we will demonstrate how to load data, run a simple
aggregation query, and get our first differentially private results. You can run
this tutorial (as well as the next ones) as you go: simply follow the
:ref:`installation instructions`, and use the copy/paste button of each code
block to reproduce it.

Throughout these tutorials, we'll imagine we are the data protection officer for
a fictional institution, the Pierre-Simon Laplace Public Library. We want to
publish statistics about how our library serves the needs of its community. Of
course, we have the privacy of our members at heart, so we want to make sure
that the data we release does not reveal anything about specific people.

This is a perfect use case for `differential privacy`_: it will allow us to
publish useful insights about groups, while protecting data about individuals.
Importantly, Tumult Analytics does *not* require you to have an in-depth
understanding of differential privacy. In these tutorials, we will gloss over
all the details of what happens behind the scenes, and focus on how to
accomplish common tasks. To learn more about the trade-offs involved in
parameter setting and mechanism design, you can consult our :ref:`Topic Guides`.

.. _differential privacy: https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html

Setup
-----

First, let's import some Python packages.

.. testcode::

   import os
   from pyspark import SparkFiles
   from pyspark.sql import SparkSession
   from tmlt.analytics.privacy_budget import PureDPBudget
   from tmlt.analytics.query_builder import QueryBuilder
   from tmlt.analytics.session import Session

Next, we initialize the Spark session.

.. _Java 11 configuration example:

.. testcode::

   spark = SparkSession.builder.getOrCreate()


.. note::

   When using Java 11, some additional configuration must be passed to Spark, so the previous code block would instead be:

   .. code-block::

      spark = (
          SparkSession.builder
          .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
          .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
          .getOrCreate()
      )

Then, we need to load our first dataset, containing information about the
members of our public library. Here, we get the data from a public ``s3``
repository, and load it into a :class:`Spark DataFrame <pyspark.sql.DataFrame>`.

.. testcode::

   spark.sparkContext.addFile(
       "https://tumult-public.s3.amazonaws.com/library-members.csv"
   )
   members_df = spark.read.csv(
       SparkFiles.get("library-members.csv"), header=True, inferSchema=True
   )

For more information about loading data files into Spark, see the Spark `data sources documentation`_.

.. _data sources documentation: https://spark.apache.org/docs/latest/sql-data-sources.html

Creating a Session
------------------

To compute queries using Tumult Analytics, we must first encapsulate the data
in a :class:`Session<tmlt.analytics.session.Session>`. The following snippet
instantiates a Session on a Spark DataFrame with our private data, using the
:meth:`from_dataframe<tmlt.analytics.session.Session.from_dataframe>` method.

.. testcode::

   session = Session.from_dataframe(
       privacy_budget=PureDPBudget(3),
       source_id="members",
       dataframe=members_df
   )

Note that in addition to the data itself, we needed to provide the
Session builder with a couple of additional pieces of information.

- The ``privacy_budget`` specifies what privacy guarantee this Session will
  provide. We will discuss this in more detail in the next tutorial.
- The ``source_id`` is the identifier for the DataFrame. We will then use it to
  refer to this DataFrame when constructing queries.

For a more complete description of the various ways a Session can be
initialized, you can consult the relevant :ref:`topic guide<Working with Sessions>`.

Evaluating queries in a Session
-------------------------------

Now that we have our Session, we can ask our first query. How many members does
our library have? To answer this question with a query, we will use the
:class:`QueryBuilder<tmlt.analytics.query_builder.QueryBuilder>` interface.

.. testcode::

   count_query = QueryBuilder("members").count()

The first part, ``QueryBuilder("members")``, specifies which private data we
want to run the query on; this corresponds to the ``source_id`` parameter from
earlier. Then, the ``count()`` statement requests the total number of records in
the dataset.

After creating our query, we need to actually run it on the data, using the
:meth:`evaluate<tmlt.analytics.session.Session.evaluate>` method of our Session.
This requires us to allocate some privacy budget to this evaluation: here, let's
evaluate the query with differential privacy, using ε=1.

.. testcode::

   total_count = session.evaluate(
       count_query,
       privacy_budget=PureDPBudget(epsilon=1)
   )

The results of the query are returned as a Spark DataFrame. We can see them
using the ``show()`` method of this DataFrame.

.. testcode::

   total_count.show()

.. testoutput::
   :hide:
   :options: +NORMALIZE_WHITESPACE

   +-----+
   |count|
   +-----+
   |...|
   +-----+

.. code-block::

   +-----+
   |count|
   +-----+
   |54215|
   +-----+

If you're running this code along with the tutorial, you might see different
values! This is a central characteristic of differential privacy: it injects
some randomization (we call this *noise*) in the execution of the query. Let's
evaluate the same query again to demonstrate this.

.. testcode::

   total_count = session.evaluate(
       count_query,
       privacy_budget=PureDPBudget(1)
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

.. code-block::

   +-----+
   |count|
   +-----+
   |54218|
   +-----+

The query result is slightly different from the previous one.

The noise added to the computation of the query can depend on the privacy
parameters, the type of aggregation, and the data itself. But in many cases, the
result will still convey accurate insights about the original data. Here, that's
the case: we can verify this by running a count query directly on the original
dataframe, which gives us the true result.

.. testcode::

   total_count = members_df.count()
   print(total_count)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   54217

We just evaluated our first differentially private query using Tumult Analytics.
In the next tutorial, we'll say a bit more about how privacy budgets work in
practice, and evaluate some more complicated queries.
