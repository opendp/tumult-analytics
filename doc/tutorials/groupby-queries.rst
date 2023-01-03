.. _Group-by queries:

Tutorial 4: Group-by queries
============================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

In all previous tutorials, all aggregations we saw were global aggregations,
returning a single statistic for all the data. But many common data analysis
operations are *group-by queries*: they partition the data into groups, and
compute one aggregation per group. In this tutorial, we will demonstrate how to
express such queries using Tumult Analytics.

In a traditional query engine, you can simply group by a column. Suppose, for
example, that you want to see the distribution of ages across the most senior
members of our library. You could count the number of members for each age, for
age 80 and above. In a language like SQL, this is likely how you would express
this query.

.. code-block::

   SELECT age, COUNT(id)
   FROM members
   WHERE age >= 80
   GROUP BY age

The result of such a query could be plotted as a bar chart looking like this.

.. image:: ../images/chart_senior_counts_1.png
    :alt: A bar chart counting the number of senior library members for each age 80 or over. The bars get progressively lower from 80 to 95.
    :align: center

Now, suppose that the oldest person living in Durham becomes a member of our
library. Assume that this person is aged 104, and that they are the only person
that age in the area. Computing the same query might get us
results that look like this.

.. image:: ../images/chart_senior_counts_2.png
    :alt: The same chart as before, except there is now a new value on the x-axis: after 95, a small bar associated with age 104 is visible.
    :align: center

There is a crucial difference between these two charts: an entirely new age
value, 104, appears in the second chart but not the first one. This is a problem
if we want to compute this query using differential privacy.

Remember: the goal of differential privacy is to avoid leaking information about
individuals. But even if we add noise to the *value* of each count, the second
chart will be enough for someone to deduce that *someone* aged 104 is a member
of the library. Otherwise, this value wouldn't appear at all! If there is a
single person in the area with this age, then it is obvious that they are a
member. We have learned precise information about a single individual, which is
supposed to be impossible.

For this reason, when computing a group-by query with differential privacy, we
will need to indicate the list of possible *group-by keys*: the different
categories that we are going to partition the data into. In the rest of this
tutorial, we will demonstrate how to do this with Tumult Analytics.

Setup
-----

Let us follow the same setup process as in the earlier tutorials, including the
infinite privacy budget—not to be used in production!

.. testcode::

   import os
   from pyspark import SparkFiles
   from pyspark.sql import SparkSession
   from tmlt.analytics.keyset import KeySet
   from tmlt.analytics.privacy_budget import PureDPBudget
   from tmlt.analytics.protected_change import AddOneRow
   from tmlt.analytics.query_builder import QueryBuilder
   from tmlt.analytics.session import Session

   spark = SparkSession.builder.getOrCreate()
   spark.sparkContext.addFile(
       "https://tumult-public.s3.amazonaws.com/library-members.csv"
   )
   members_df = spark.read.csv(
       SparkFiles.get("library-members.csv"), header=True, inferSchema=True
   )

   session = Session.from_dataframe(
       privacy_budget=PureDPBudget(epsilon=float('inf')),
       source_id="members",
       dataframe=members_df,
       protected_change=AddOneRow(),
   )

Introduction to KeySets
-----------------------

To specify the list of group-by keys in Tumult Analytics, we use the
:class:`tmlt.analytics.keyset.KeySet` class. A KeySet specifies both the columns
by which we are going to group by, and the possible values for those columns.

The simple way to initialize a KeySet, especially when there are only a few
possible values for a given column, is to use
:meth:`~tmlt.analytics.keyset.KeySet.from_dict`. For example, the following
KeySet enumerates all possible values for the categorical column
`education_level`.

.. testcode::

   edu_levels = KeySet.from_dict({
       "education_level": [
           "up-to-high-school",
           "high-school-diploma",
           "bachelors-associate",
           "masters-degree",
           "doctorate-professional",
       ]
   })

Once we have this KeySet, we can use it in group-by queries, using the
:meth:`~tmlt.analytics.query_builder.QueryBuilder.groupby` operation. For
example, let us compute the average age of library members, grouped by education
level.

.. testcode::

   edu_average_age_query = (
       QueryBuilder("members")
       .groupby(edu_levels)
       .average("age", low=0, high=120)
   )
   edu_average_ages = session.evaluate(
       edu_average_age_query,
       privacy_budget=PureDPBudget(1),
   )
   edu_average_ages.sort("age_average").show(truncate=False)

.. testoutput::
   :hide:
   :options: +NORMALIZE_WHITESPACE

   +----------------------+...+
   |education_level       |age_average |
   +----------------------+...+
   |up-to-high-school     |...|
   |high-school-diploma   |...|
   |bachelors-associate   |...|
   |masters-degree        |...|
   |doctorate-professional|...|
   +----------------------+...+

.. code-block::

   +----------------------+-----------+
   |education_level       |age_average|
   +----------------------+-----------+
   |up-to-high-school     |18.00410415|
   |high-school-diploma   |43.68196862|
   |bachelors-associate   |46.27907318|
   |masters-degree        |49.70756023|
   |doctorate-professional|51.71076923|
   +----------------------+-----------+

The same data can be represented graphically using your favorite visualization
tool. For example, the following uses `seaborn <https://seaborn.pydata.org/>`__;
if you want to run it locally, you can install it with ``pip install seaborn``.

.. code-block::

   import matplotlib.pyplot as plt
   import seaborn as sns

   sns.set_theme(style="whitegrid")
   g = sns.barplot(
       x="education_level",
       y="age_average",
       data=edu_average_ages.toPandas().sort_values("age_average"),
       color="#1f77b4",
   )
   g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")
   plt.title("Average age of library members, by education level")
   plt.xlabel("Education level")
   plt.ylabel("Average age")
   plt.tight_layout()
   plt.show()

.. image:: ../images/chart_average_age_by_edu.png
    :alt: A bar chart plotting the average age of library members, by education level. The first bar is small, at about 18 for "up-to-high-school", then the bars increase from approximately 43 to approximately 52 as education level increases.
    :align: center

A value in a KeySet will appear in the output, and a value that is not in a
KeySet will not, regardless of which values appear in the actual data. For
example, in our fake dataset, all the age values are 6 or above: younger
children cannot be members of our library. So, what happens if we compute counts
for age values between *5* and 17?

.. testcode::

   young_ages = list(range(5, 18))  # [5, 6, ..., 17]
   young_age_keys = KeySet.from_dict({"age": young_ages})
   young_age_query = (
       QueryBuilder("members")
       .groupby(young_age_keys)
       .count()
   )
   young_age_counts = session.evaluate(
       young_age_query,
       PureDPBudget(0.1)
   )

.. code-block::

    sns.barplot(
        x="age",
        y="count",
        data=young_age_counts.toPandas().sort_values("age"),
        color="#1f77b4",
    )
    plt.title("Count of younger library members, by age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

.. image:: ../images/chart_younger_age_counts.png
    :alt: A bar chart plotting the count of library members for ages 5 to 17. All the values are about 100, except for age 5, where it is about 20.
    :align: center

We observe a low, but non-zero count for age 5, even though this value is
completely absent in our dataset. This is entirely due to the noise added to the
real value (here, 0).

Multiple columns
----------------

So far, we saw how to run group-by queries, where we grouped by a single column.
What if we want to group by multiple columns? One simple way is to use a Python
dictionary with multiple values. Let's take an example, and compute counts by
age (of teenagers and young adults) and education level.

.. testcode::

   teen_edu_keys = KeySet.from_dict({
       "age": list(range(13, 22)), # [13, 14, ..., 21]
       "education_level": [
           "up-to-high-school",
           "high-school-diploma",
           "bachelors-associate",
           "masters-degree",
           "doctorate-professional",
       ],
   })

This gives us a KeySet with each combination of values across the two columns
``age`` and ``education_level``. To manually check what's inside of a KeySet,
we can call its :meth:`tmlt.analytics.keyset.KeySet.dataframe` method, which
will return the group-by keys it encapsulates, as a Spark DataFrame.

.. testcode::

   teen_edu_df = teen_edu_keys.dataframe()
   teen_edu_df.sort("age", "education_level").show(n=12, truncate=False)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   +---+----------------------+
   |age|education_level       |
   +---+----------------------+
   |13 |bachelors-associate   |
   |13 |doctorate-professional|
   |13 |high-school-diploma   |
   |13 |masters-degree        |
   |13 |up-to-high-school     |
   |14 |bachelors-associate   |
   |14 |doctorate-professional|
   |14 |high-school-diploma   |
   |14 |masters-degree        |
   |14 |up-to-high-school     |
   |15 |bachelors-associate   |
   |15 |doctorate-professional|
   +---+----------------------+
   only showing top 12 rows

And we can use this KeySet to perform a group-by query on multiple columns.

.. testcode::

   teen_edu_query = (
       QueryBuilder("members")
       .groupby(teen_edu_keys)
       .count()
   )
   teen_edu_counts = session.evaluate(
       teen_edu_query,
       PureDPBudget(0.2),
   )

   teen_edu_counts.sort("age", "education_level").show(n=12, truncate=False)

.. testoutput::
   :hide:
   :options: +NORMALIZE_WHITESPACE

   +---+----------------------+-----+
   |age|education_level       |count|
   +---+----------------------+-----+
   |13 |bachelors-associate   |...|
   |13 |doctorate-professional|...|
   |13 |high-school-diploma   |...|
   |13 |masters-degree        |...|
   |13 |up-to-high-school     |...|
   |14 |bachelors-associate   |...|
   |14 |doctorate-professional|...|
   |14 |high-school-diploma   |...|
   |14 |masters-degree        |...|
   |14 |up-to-high-school     |...|
   |15 |bachelors-associate   |...|
   |15 |doctorate-professional|...|
   +---+----------------------+-----+
   only showing top 12 rows

.. code-block::

   +---+----------------------+-----+
   |age|education_level       |count|
   +---+----------------------+-----+
   |13 |bachelors-associate   |-7   |
   |13 |doctorate-professional|10   |
   |13 |high-school-diploma   |-1   |
   |13 |masters-degree        |1    |
   |13 |up-to-high-school     |949  |
   |14 |bachelors-associate   |-13  |
   |14 |doctorate-professional|6    |
   |14 |high-school-diploma   |2    |
   |14 |masters-degree        |-3   |
   |14 |up-to-high-school     |931  |
   |15 |bachelors-associate   |-17  |
   |15 |doctorate-professional|-8   |
   +---+----------------------+-----+
   only showing top 12 rows

Note that we get some values in the output that are very low, or even negative. This
makes sense: some combinations of ``age`` and ``education_level`` don't actually
appear in the input. Teens aged 13 don't have PhDs! So what we see for these
columns is pure noise, added to zero values.
We can fix that problem by creating a KeySet that only contains reasonable
combinations of values. One way to do this is by creating a Spark DataFrame with
the desired combinations, and initializing a KeySet with it using the
:meth:`~tmlt.analytics.keyset.KeySet.from_dataframe` method. Here, let's create
such a KeySet for our age range, making some assumptions about the minimum age at
which certain education levels can be achieved.

.. testcode::

   teen_edu_subset_keys_df = spark.createDataFrame(
       [
           (13, "up-to-high-school"),
           (14, "up-to-high-school"),
           (15, "up-to-high-school"),
           (16, "up-to-high-school"),
           (17, "up-to-high-school"),
           (17, "high-school-diploma"),
           (18, "up-to-high-school"),
           (18, "high-school-diploma"),
           (19, "up-to-high-school"),
           (19, "high-school-diploma"),
           (19, "bachelors-associate"),
           (20, "up-to-high-school"),
           (20, "high-school-diploma"),
           (20, "bachelors-associate"),
           (21, "up-to-high-school"),
           (21, "high-school-diploma"),
           (21, "bachelors-associate"),
       ],
       ["age", "education_level"],
   )
   teen_edu_subset_keys = KeySet.from_dataframe(teen_edu_subset_keys_df)

Once we have our KeySet, we can use it in the desired aggregation…

.. testcode::

    teen_edu_query = (
        QueryBuilder("members")
        .groupby(teen_edu_keys)
        .count()
    )
    teen_edu_counts = session.evaluate(
        teen_edu_query,
        PureDPBudget(0.2),
    )

… and plot the results.

.. code-block::

   edu_order = ["up-to-high-school", "high-school-diploma", "bachelors-associate"]
   sns.barplot(
       x="age",
       y="count",
       hue="education_level",
       hue_order=edu_order,
       data=teen_edu_counts.toPandas()
   )
   plt.ylim(0, 1300)
   plt.title("Count of teenage library members, by age and education level")
   plt.xlabel("Age")
   plt.ylabel("Count")
   plt.legend(loc="upper left")
   plt.show()

.. image:: ../images/chart_teen_edu_counts.png
    :alt: A bar chart plotting the count of library members, by age (for ages 13 to 21) and education level. There is only the "up-to-high-school" bar for ages 13 to 16, then "high-school-diploma" starts at 17 and becomes prominent at age 18, and "bachelors-associate" starts rising starting at age 20.
    :align: center

This is also particularly useful when there are a large number of group-by keys:
you can load a large Spark DataFrame into a KeySet directly, without having to
enumerate the values in a Python dictionary.

Manipulating KeySets
--------------------

Most applications of Tumult Analytics require specifying the keys to group-by
queries. To make this process as easy as possible, we added first-class support
for common KeySet manipulation operations. Let us review these operations.

Cross-product
"""""""""""""

When two KeySets are specifying group-by keys for different sets of columns, you
can use the
:meth:`multiplication operator<tmlt.analytics.keyset.KeySet.__mul__>` to
generate the combination of values present in both KeySets. Consider, for
example, the combination of teen ages and education levels that we started with
in the previous section.

.. testcode::

   teen_edu_keys = KeySet.from_dict({
       "age": list(range(13, 22)), # [13, 14, ..., 21]
       "education_level": [
           "up-to-high-school",
           "high-school-diploma",
           "bachelors-associate",
           "masters-degree",
           "doctorate-professional",
       ],
   })

There is another way to specify the same KeySet: initializing one KeySet for
each column, and taking the cross-product of both using the multiplication
operator.

.. testcode::

   teen_keys = KeySet.from_dict({
       "age": list(range(13, 22)), # [13, 14, ..., 21]
   })
   edu_keys = KeySet.from_dict({
       "education_level": [
           "up-to-high-school",
           "high-school-diploma",
           "bachelors-associate",
           "masters-degree",
           "doctorate-professional",
       ],
   })

   print(teen_edu_keys == teen_keys * edu_keys)

.. testoutput::

   True

Projection
""""""""""

What if you have a large KeySet containing possible combination of values
between multiple columns, and want to select only a subset of these columns? For
this use case, you can use the
:meth:`projection operator<tmlt.analytics.keyset.KeySet.__getitem__>`. For
example, taking the two-column KeySet above and projecting it on a single column
will retrieve the original one-column KeySet.

.. testcode::

   print(edu_keys == teen_edu_keys["education_level"])

.. testoutput::

   True

Note that this operation, just like the
:meth:`~tmlt.analytics.keyset.KeySet.from_dataframe` method, removes duplicates
in the data: ``teen_edu_keys["education"]`` only has one row for e.g.
``"up-to-high-school"``, even though this value appears in multiple rows in the
original KeySet.

You can easily specify multiple columns using multiple parameters, or a list.

.. testcode::

   print(teen_edu_keys == teen_edu_keys["age", "education_level"])

.. testoutput::

   True

.. testcode::

   print(teen_edu_keys == teen_edu_keys[["age", "education_level"]])

.. testoutput::

   True

Filters
"""""""

Finally, KeySets support the same :meth:`~tmlt.analytics.keyset.KeySet.filter`
operation as Spark DataFrames. For example, consider the KeySet with specific
combinations of age and education level from before.

.. testcode::

   teen_edu_subset_keys_df = spark.createDataFrame(
       [
           (13, "up-to-high-school"),
           (14, "up-to-high-school"),
           (15, "up-to-high-school"),
           (16, "up-to-high-school"),
           (17, "up-to-high-school"),
           (17, "high-school-diploma"),
           (18, "up-to-high-school"),
           (18, "high-school-diploma"),
           (19, "up-to-high-school"),
           (19, "high-school-diploma"),
           (19, "bachelors-associate"),
           (20, "up-to-high-school"),
           (20, "high-school-diploma"),
           (20, "bachelors-associate"),
           (21, "up-to-high-school"),
           (21, "high-school-diploma"),
           (21, "bachelors-associate"),
       ],
       ["age", "education_level"],
   )
   teen_edu_subset_keys = KeySet.from_dataframe(teen_edu_subset_keys_df)

We could also have generated this KeySet using our larger KeySet with all
possible combinations of the two columns, using a filter.

.. testcode::

   teen_edu_keys_filtered = teen_edu_keys.filter("""
       education_level != 'masters-degree'
       AND education_level != 'doctorate-professional'
       AND NOT (education_level = 'bachelors-associate' AND age < 19)
       AND NOT (education_level = 'high-school-diploma' AND age < 17)
   """)
   print(teen_edu_keys_filtered == teen_edu_subset_keys)

.. testoutput::

   True
