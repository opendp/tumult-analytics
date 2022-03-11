Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`__.

0.1.1 - 2022-02-28
------------------

Added
~~~~~

-  Added a ``KeySet`` class, which will eventually be used for all GroupBy queries.
-  Added ``QueryBuilder.groupby()``, a new group-by based on ``KeySet``\ s.

Changed
~~~~~~~

-  The Analytics library now uses ``KeySet`` and ``QueryBuilder.groupby()`` for all
   GroupBy queries.
-  The various ``Session`` methods for loading in data from CSV no longer support loading the data’s schema from a file.

Deprecated
~~~~~~~~~~

-  ``QueryBuilder.groupby_domains()`` and ``QueryBuilder.groupby_public_source()`` are now deprecated in favor of using ``QueryBuilder.groupby()`` with ``KeySet``\ s.
   They will be removed in a future version.

.. _section-1:

0.1.0 - 2022-02-15
------------------

.. _added-1:

Added
~~~~~

-  Initial release
