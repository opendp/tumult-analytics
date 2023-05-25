Using Tumult Analytics on Databricks
====================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

This guide covers installing and running Tumult Analytics in notebooks on `Databricks <https://www.databricks.com/>`__.
It assumes that you have an existing Databricks workspace with a compatible compute cluster available.
Tumult Analytics is currently compatible with versions 7.3 LTS through 12.2 LTS of the `Databricks runtime <https://docs.databricks.com/release-notes/runtime/releases.html>`__.

Installation
^^^^^^^^^^^^

Python packages can be installed on Databricks in two ways, either on a per-notebook level or at the cluster level.
To get started, let's install Tumult Analytics in a notebook.

Create a new notebook, and connect it to a cluster using any supported version of the Databricks runtime.
Installing packages is straightforward using the |%pip directive|_: add

.. code-block::

   %pip install tmlt.analytics

as the first code block in the notebook, and execute it.
This will install Tumult Analytics and all of its dependencies, but only for use in this notebook.
It will need to be reinstalled each time the notebook is connected to a cluster, or the cluster is rebooted.

.. |%pip directive| replace:: ``%pip`` directive
.. _%pip directive: https://docs.databricks.com/libraries/notebooks-python-libraries.html#manage-libraries-with-pip-commands

Alternatively, packages can be installed on a compute cluster, automatically making them available to all notebooks using that cluster.
This is likely the more convenient way to install packages that will be used frequently.
To install Tumult Analytics this way, navigate to the Databricks compute view in the left sidebar.
Select the cluster you wish to modify, go to the "Libraries" tab, and click "Install new".
Use "PyPI" as the library source, enter ``tmlt.analytics`` as the package, and hit "Install".
After a few moments, the package will show as installed, and you can begin using Tumult Analytics in any notebooks attached to this cluster.
Python version specifiers, e.g. ``tmlt.analytics>=0.7.0,<0.8``, are also supported if you wish to install a particular version.

Regardless of which installation process you used, the installation check described in the :ref:`local installation instructions <Installation instructions>` can be run by adding a new cell to your notebook with the following content:

.. code-block::

   from tmlt.analytics.utils import check_installation
   check_installation()

and then running it.
If you encounter an error at this stage, please `let us know <https://gitlab.com/tumult-labs/analytics/-/issues>`__!


Loading and Saving Data in Databricks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a query with Tumult Analytics in a Databricks notebook is essentially identical to doing so anywhere else, the only difference being how tables are loaded and saved.
To begin, let's initialize a Databricks table by loading a dataset from S3 and saving it as a table.

.. code-block::

   spark.sql('CREATE DATABASE IF NOT EXISTS main.library;')
   spark.sql('CREATE DATABASE IF NOT EXISTS main.library_output;')
   df = spark.read.csv("s3://tumult-public/library-members.csv", header=True, inferSchema=True)
   df.write.saveAsTable("main.library.members")

This creates two new Databricks schemas in the default ``main`` catalog, ``library`` and ``library_output``, which are used respectively for inputs and outputs in this guide.
It then writes the library members dataset from our :ref:`tutorials <Tutorials>` into Databricks as the ``members`` table in the ``library`` schema.

.. note::

   Specifying the catalog to be used in this way is only supported in environments and on versions of the Databricks runtime that support `Unity Catalog <https://docs.databricks.com/data-governance/unity-catalog/index.html>`__.
   Otherwise, only the schema and table name may be given.
   For example, with versions of the Databricks runtime prior to 10 the above code block would instead be:

   .. code-block::

      spark.sql('CREATE DATABASE IF NOT EXISTS library;')
      spark.sql('CREATE DATABASE IF NOT EXISTS library_output;')
      df = spark.read.csv("s3://tumult-public/library-members.csv", header=True, inferSchema=True)
      df.write.saveAsTable("library.members")

   In this case, the ``main`` catalog would be used by default.

Once this is done, the :meth:`spark.table.read <pyspark.sql.DataFrameReader.table>` function can be used to read the data from Databricks, for example:

.. code-block::

   df = spark.read.table('main.library.members')

Here, we are loading the library members dataset that we just saved to Databricks, but if the table was already in Databricks this read would allow accessing it.

.. note::

   As with ``saveAsTable`` above, specifying the catalog to read from is only supported when using Unity Catalog.
   If you are not, pass only the schema and table name (e.g. ``library.members``).

Once this dataset is loaded, queries can be evaluated as normal.
For this example, let's just count how many users have borrowed a book:

.. code-block::

   from tmlt.analytics.privacy_budget import PureDPBudget
   from tmlt.analytics.protected_change import AddOneRow
   from tmlt.analytics.session import Session
   from tmlt.analytics.query_builder import QueryBuilder

   budget = PureDPBudget(1)
   sess = Session.from_dataframe(
       privacy_budget=budget,
       source_id="members",
       dataframe=df,
       protected_change=AddOneRow(),
   )
   output = sess.evaluate(
       QueryBuilder("members").filter('books_borrowed > 0').count(),
       budget
   )

Finally, we need to write out this data so that it can be used elsewhere, which works the same as saving the input data did above:

.. code-block::

   output.write.saveAsTable('main.library_output.active_members')

This writes out the result of our query to the ``active_members`` table in the ``output`` schema of the ``main`` catalog.

Because Databricks `does not run <https://docs.databricks.com/libraries/index.html>`__ ``atexit`` functions, Analytics may leave behind some temporary tables.
These can be removed by running the :func:`~tmlt.analytics.utils.cleanup` function when you are done with each notebook session:

.. code-block::

   from tmlt.analytics.utils import cleanup
   cleanup()

If many such temporary tables have accumulated over time, the :func:`~tmlt.analytics.utils.remove_all_temp_tables` function can be used to clean them all up, though this may erase query results for any active notebooks if they haven't been saved elsewhere.

As you can see, using Tumult Analytics on Databricks is very straightforward.
If you encounter any unexpected issues, please let us know by `filing an issue <https://gitlab.com/tumult-labs/analytics/-/issues>`__ or reaching out on our `Slack server <https://tmlt.dev/slack>`__.
