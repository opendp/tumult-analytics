.. _Privacy budget fundamentals:


Privacy budget fundamentals
===========================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

This topic guide goes into more depth on the concept of *privacy budgets* that
we discussed in :ref:`tutorial two <Working with privacy budgets>`. At a high
level, a privacy budget associates one or more numeric *privacy parameters* with
a *privacy definition*. Together, this information determines the privacy
guarantees provided by each query, and by the Session as a whole.

In particular, a privacy budget determines how much randomization is added
to the computation of a query. Choosing an appropriate budget
for a given query is ultimately an exercise in balancing privacy and accuracy
needs; we discuss this topic more :ref:`below <Privacy tradeoff>`.

Tumult Analytics currently supports two distinct privacy definitions:

* *Pure Differential Privacy (Pure DP)*, with its associated privacy parameter ``epsilon``.
  For data publication use cases, the value of ``epsilon`` is often chosen to be
  `lower than 5 <https://desfontain.es/privacy/real-world-differential-privacy.html>`_.
  Values below 1 are typically considered conservative.
* *Zero Concentrated Differential Privacy (zCDP)*, with its associated
  privacy parameter ``rho``.

For both pure DP and zCDP, a higher budget
leads to less randomization and thus more accurate results, whereas
a lower budget yields more noisy results.

Understanding the differences between these two privacy definitions is out of scope
for this guide; for simple use cases, pure DP is a good default choice.
To learn more, you can consult this `blog post <https://desfontain.es/privacy/renyi-dp-zero-concentrated-dp.html>`__.

Using privacy budgets in Tumult Analytics
-----------------------------------------

Tumult Analytics provides the :class:`~tmlt.analytics.privacy_budget.PrivacyBudget`
interface and two concrete implementations for specifying privacy budgets,
depending on which privacy definition you wish to use:
:class:`~tmlt.analytics.privacy_budget.PureDPBudget`
and :class:`~tmlt.analytics.privacy_budget.RhoZCDPBudget`.

Recall from tutorials :ref:`one <First steps>` and
:ref:`two <Working with privacy budgets>` that when you initialize a Session,
you must allocate a total privacy budget for it. Then, each time you evaluate a
query through the Session, you must specify how much budget the query should
use, which is then subtracted from the Session's total. For example, if you
initialize a Session with :code:`PureDPBudget(epsilon=5)`, and then evaluate a
query with :code:`PureDPBudget(epsilon=2)`, then your Session's remaining budget
will be :code:`PureDPBudget(epsilon=3)`.

There are a few additional constraints worth noting:

* The type of budget your queries use must match the type of budget the
  Session was initialized with; for example, if you initialize your Session with a
  :code:`PureDPBudget`, then each query evaluated through that Session
  must also use a :code:`PureDPBudget`.
* The individual budgets requested by all of your queries combined
  may not exceed the Session's total budget.
  If you attempt to evaluate a query with a greater budget than your Session has
  remaining, the Session will raise an exception.


.. _Privacy tradeoff:

Choosing a budget: privacy vs. accuracy
---------------------------------------

When you're deciding how much privacy budget to use for a query, there is no single
"right" choice. The choice of budget
is ultimately a balance between accuracy and privacy, with lower budgets
yielding noisier results and higher budgets yielding more accurate
(and thus less private) results.

To better understand how epsilon impacts this tradeoff, let's look at a simple
example. We'll use the example dataset from :ref:`tutorial one <First steps>`,
and we'll perform a simple query to count the number of records in the dataset.
The figure below plots the results of running this count query using a
:code:`PureDPBudget` with 3 different epsilon values, 50 times each:

.. image:: ../images/chart_counts_different_eps.png
    :scale: 100%
    :alt: Graph plotting noisy counts under different epsilon values. The values are
            further away from the mean when using a small epsilon (here, 0.2) than
            when using a large epsilon (here, 1.0).
    :align: center

Notice how a smaller budget results in a larger spread of output values, while a
larger budget results in a tighter cluster of values with less noise.
The result of any individual query evaluated using a larger budget is more likely
to resemble the true answer, and thus a larger budget provides less privacy protection
than a smaller budget.

Understanding the total privacy guarantee of a Session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The more budget you allocate to your Session, the more you will be able to use in each
individual query (or you can ask a larger number of queries, each with a smaller
budget). However, with a larger aggregate budget, the total privacy guarantee of the
Session gets worse.

To understand why this is the case, let's describe the attacker model more explicitly.
Suppose you use a Session to generate answers to queries on a database,
while enforcing a given privacy budget on a set of queries. An attacker is trying
to determine whether a specific record (their target) is present in the input database.
This attacker is powerful: we assume that they know all the records in the database,
except their target.

Suppose the attacker starts with a uniform prior suspicion about whether their target is
in the database (i.e., an initial certainty of 50%, or 0.5).
Next, they access the output of differentially private queries that someone previously
published from the database.
The choice of epsilon used for these queries determines how much the attacker's
suspicion can change. The below graph plots an attacker's
maximum updated certainty against various epsilon values:

.. image:: ../images/chart_attacker_certainty.png
    :scale: 100%
    :alt: Graph plotting an attacker's maximum updated certainty regarding whether
            their target is in the database or not, given a prior suspicion of 50%.
            Increasing epsilon from 0 to 3 allows the attacker to significantly
            improve their certainty (form 50% to around 90%), and further increasing
            epsilon quickly approaches 100% certainty.
    :align: center

Recall that smaller epsilon values introduce more noise into the output of
differentially private queries.
Therefore, smaller epsilon values do not allow the attacker to significantly update
their suspicion, whereas larger epsilons allow the attacker to determine with increasing
certainty whether or not their target is in the database.

For an even more in-depth explanation of this topic, you can check out the following
`blog post <https://desfontain.es/privacy/differential-privacy-in-more-detail.html>`__.

The impact of data size
^^^^^^^^^^^^^^^^^^^^^^^

Another factor that impacts the privacy/accuracy tradeoff associated with a given
budget is the size of each group on which aggregations are computed. In our first example
above, even with the smallest budget of 0.2, all the noisy results were within
about +/- 25 of the true count, which is a relative spread of about 0.05%.
But what happens if we aggregate the data in smaller groups?
Consider again our database of library members. Instead of counting all records
in the database, we'll first group members by age and education level, and then
count how many members fall in each group. The below graph plots the percentage error
between the true and noisy counts for each group. The noisy counts were computed
using an epsilon of 0.2.

.. image:: ../images/chart_error_vs_partition_age_edu.png
    :scale: 80%
    :alt: Graph plotting the error percentage of a noisy count query against the size
            of the partition the query was evaluated on. Here, each partition represents
            one combination of age x education_level for members in our example database.
            The same epsilon value (here, 0.2) applied to smaller partitions yields
            dramatically more relative error than when applied to larger partitions.
    :align: center

Note the log scales for both x and y axes. For any given epsilon, queries evaluated on
larger groups will tend to have less error than when evaluated on smaller groups.
It is important to consider the typical sizes of groups of interest in your data
when determining an appropriate privacy budget.

