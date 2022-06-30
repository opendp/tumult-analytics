.. _Spark:

Spark
=====

Tumult Analytics uses Spark as its underlying data processing
framework. This topic guide covers relevant information about Spark
for users of the Analytics library.

Configuring Spark sessions
--------------------------

Analytics uses :class:`Spark sessions <pyspark.sql.SparkSession>` to do all data processing operations.
As long as Spark has an active session, any calls that "create" new Spark
sessions use that active session.

If you want Analytics to use Spark sessions with any *specific* properties,
then before running Analytics code, you should create that Spark session:

.. code-block::

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.<your options here>.getOrCreate()

As long as this session is active, Analytics will use it.

Connecting to Hive
^^^^^^^^^^^^^^^^^^

If you want to connect Spark to an existing `Hive <https://hive.apache.org/>`_
database, you should use the following options when creating a Spark session:

.. code-block::

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.<your options here>
        .config('spark.sql.warehouse.dir', '<Hive warehouse directory>')
        .enableHiveSupport()
        .getOrCreate()

To see where Hive's warehouse directory is, you can use the
`Hive CLI <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Cli#LanguageManualCli-HiveInteractiveShellCommands>`_
(or its replacement,
`Beehive <https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients#HiveServer2Clients-BeelineHiveCommands>`_)
to view the
`relevant configuration parameter <https://cwiki.apache.org/confluence/display/Hive/AdminManual+Metastore+3.0+Administration#AdminManualMetastore3.0Administration-GeneralConfiguration>`_:

.. code-block::

        > set hive.metastore.warehouse.dir;
        hive.metastore.warehouse.dir=/hive/warehouse

Materialization and data cleanup
--------------------------------

Tumult Analytics is built on top of Tumult Core, which
implements all of the differential privacy primitives that Analytics uses.
Tumult Core uses a Spark database (named "``tumult_temp_<time>_<uuid>``") to
materialize dataframes after noise has been added. This ensures that repeated
queries on a dataframe of results do not re-evaluate the query with fresh
randomness.

This has a few consequences for users:

* Queries are eagerly-evaluated, instead of lazily-evaluated.
* Operations create a temporary database in Spark.
* Analytics *does not* support multi-threaded operations, because the
  materialization step changes the active Spark database. (The active database is
  changed back to the original database at the end of the materialization step.)

Automatically cleaning up temporary data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tumult Core registers a cleanup function with ``atexit``
(see `Python's atexit documentation <https://docs.python.org/3/library/atexit.html>`_).
If a Spark session is still active when the program exits normally, this cleanup
function will automatically delete the materialization database.

If you wish to call ``spark.stop()`` before program exit, you should call
:func:`~tmlt.analytics.cleanup.cleanup()` first. This will delete the materialization
database. This function requires an active Spark session, but is otherwise safe
to call at any time in a single-threaded program. (If
:func:`~tmlt.analytics.cleanup.cleanup()` is called before a materialization step,
Core will create a new materialization database.)

Finding and removing leftover temporary data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The materialization database is stored as a folder in your Spark
warehouse directory.  If your program exits unexpectedly (for example,
because it was terminated with Ctrl-C),
or if the cleanup function is called without an active Spark session,
this temporary database (and its associated folder) may not be deleted.

Analytics has a function to delete any of these folders in the current
Spark warehouse: :func:`~tmlt.analytics.cleanup.remove_all_temp_tables`.
As long as your program is single-threaded, it is safe to call this function
at any time.

You can also manually delete this database by deleting its
directory from your Spark warehouse directory.
(If you did not explicitly configure a Spark warehouse directory,
look for a directory called ``spark-warehouse``.)
Spark represents databases as folders; the databases used
for materialization will be folders named "``tumult_temp_<time>_<uuid>``".
Deleting the folder will delete the database.

These folders are safe to manually delete any time that your program is not running.

Performance and profiling
-------------------------

All queries made with Analytics are executed by Spark. If you are having
performance problems, you will probably want to look at
`Spark performance-tuning options <https://spark.apache.org/docs/latest/sql-performance-tuning.html>`_.
