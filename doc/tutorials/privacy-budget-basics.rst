.. _Working with privacy budgets:

Tutorial 2: Working with privacy budgets
========================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023


In our :ref:`first tutorial<First steps>`, we saw how to run a simple aggregation
query on private data. When loading the private data into a Session, we had to
specify a *privacy budget*. This raises two kinds of questions.

1. What is a privacy budget, what formal guarantee does it provide, and how can
   we choose the value of this parameter?
2. How do we work with privacy budgets using Tumult Analytics, and what is the
   privacy promise of the interface?

In this tutorial, we will focus on the second question; the only thing you will
need to know is that a *smaller* privacy budget translates to a *stronger*
privacy guarantee. If you want to first learn more about privacy budget
fundamentals, you can consult the following resources.

- If you're interested in understanding the formal guarantee of privacy budgets,
  you can consult this `explainer`_. It presents an intuitive interpretation of
  differential privacy using *betting odds*, and formalizes it using a Bayesian
  attacker model.
- If you would like to know what privacy parameters are commonly used for data
  publication, you can consult this `list of real-world use cases`_.

.. _explainer: https://desfontain.es/privacy/differential-privacy-in-more-detail.html

.. _list of real-world use cases: https://desfontain.es/privacy/real-world-differential-privacy.html

These are only optional reading! The one-sentence summary above (smaller budget
= better privacy) is enough to follow the rest of this tutorial. Let's get
started!

Setup
-----

Just like earlier, we import Python packages...

.. testcode::

   import os
   from pyspark import SparkFiles
   from pyspark.sql import SparkSession
   from tmlt.analytics.privacy_budget import PureDPBudget
   from tmlt.analytics.protected_change import AddOneRow
   from tmlt.analytics.query_builder import QueryBuilder
   from tmlt.analytics.session import Session


... and download the dataset, in case we haven't already done so.

.. testcode::

   spark = SparkSession.builder.getOrCreate()
   spark.sparkContext.addFile(
       "https://tumult-public.s3.amazonaws.com/library-members.csv"
   )
   members_df = spark.read.csv(
       SparkFiles.get("library-members.csv"), header=True, inferSchema=True
   )

Creating a Session with a fixed budget
--------------------------------------

Let's initialize our :class:`Session<tmlt.analytics.session.Session>`. We will
allocate a fixed privacy budget of ``epsilon=2.5`` to it, using the classical
("pure") differential privacy definition.

.. testcode::

   budget = PureDPBudget(epsilon=2.5) # maximum budget consumed in the Session
   session = Session.from_dataframe(
       privacy_budget=budget,
       source_id="members",
       dataframe=members_df,
       protected_change=AddOneRow(),
   )

Let's check that we initialized the Session as intended using the
:meth:`describe<tmlt.analytics.session.Session.describe>` method:

.. testcode::

   session.describe()

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   The session has a remaining privacy budget of PureDPBudget(epsilon=2.5).
   The following private tables are available:
   Table 'members' (no constraints):
      Columns:
         - 'id'               INTEGER
         - 'name'             VARCHAR
         - 'age'              INTEGER
         - 'gender'           VARCHAR
         - 'education_level'  VARCHAR
         - 'zip_code'         VARCHAR
         - 'books_borrowed'   INTEGER
         - 'favorite_genres'  VARCHAR
         - 'date_joined'      TIMESTAMP

Initializing a Session with a finite privacy budget gives a simple interface
promise: all queries evaluated on this Session, *taken together*, will provide
differentially private results with at most ``epsilon=2.5``. This parameter
measures the potential privacy *loss*: a lower epsilon gives a stricter limit on
the privacy loss, and therefore a higher level of protection. Here, the
corresponding interface promise is a *privacy guarantee*: it enforces a minimum
level of protection on the private data. For more information about this promise
and its caveats, you can consult the :ref:`relevant topic guide<Privacy
promise>`.

Now, how does the Session enforce that guarantee in practice?

Consuming the budget by evaluating queries
------------------------------------------

Each time we evaluate a query in our Session, we will *consume* some of the
overall budget, and we will need to specify *how much* of this budget we want to
consume. Let's start with a simple example: how many minors are members of the
library? We will answer that question using a simple
:meth:`filter<tmlt.analytics.query_builder.QueryBuilder.filter>` query,
consuming ``epsilon=1`` out of our total budget.

.. testcode::

   minor_query = QueryBuilder("members").filter("age < 18").count()
   minor_count = session.evaluate(
       minor_query,
       privacy_budget=PureDPBudget(epsilon=1),
   )
   minor_count.show()

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
   |13817|
   +-----+

Now, evaluating that query *consumed* some of our privacy budget. To see this,
we can consult the Session's ``remaining_privacy_budget``:

..
    TODO(#1642): It makes absolutely zero sense that the above is needed for the
    tests to pass.

.. testcode::

   print(session.remaining_privacy_budget)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   PureDPBudget(epsilon=1.5)

We consumed a budget of 1 out of a total of 2.5, so there is 1.5 left. Let's try
another query: how many library members have a Master's degree or a higher level
of formal education?

.. testcode::

   edu_query = (
       QueryBuilder("members")
       .filter("education_level IN ('masters-degree', 'doctorate-professional')")
       .count()
   )
   edu_count = session.evaluate(
       edu_query,
       privacy_budget=PureDPBudget(epsilon=1),
   )
   edu_count.show()

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
   | 4765|
   +-----+

You can probably guess how much budget we have left:

.. testcode::

   print(session.remaining_privacy_budget)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   PureDPBudget(epsilon=0.5)

Now, what happens if we try to consume *more* budget than what we have left?

.. testcode::

   total_count = session.evaluate(
       QueryBuilder("members").count(),
       privacy_budget=PureDPBudget(epsilon=1),
   )

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   Traceback (most recent call last):
   RuntimeError: Cannot answer query without exceeding the Session privacy budget.
   Requested: ε=1.000
   Remaining: ε=0.500
   Difference: ε=0.500

The ``evaluate`` call returns an error. This is how the Session enforces its
privacy promise: it makes sure that the queries cannot consume more than the
initial privacy budget.

Note that since the call to ``evaluate`` was rejected by the Session, it did not
consume any privacy budget.

.. testcode::

   print(session.remaining_privacy_budget)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   PureDPBudget(epsilon=0.5)

If we don't consume this leftover budget, that's OK: the privacy promise is
still enforced. But of course, this is somewhat "wasteful": we could have
answered more queries, or allocated more budget to answer previous queries more
accurately. Here, let us simply modify the last query to use all the budget that
we have left.

.. testcode::

   total_count = session.evaluate(
       QueryBuilder("members").count(),
       privacy_budget=session.remaining_privacy_budget,
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
   |54215|
   +-----+

Now, suppose you have a fixed privacy budget, and your task is to publish the
result of multiple queries. How to split the privacy budget across the different
queries? To learn more about this question, you can consult our longer
:ref:`topic guide <Privacy Budget Fundamentals>` about privacy budget fundamentals.
