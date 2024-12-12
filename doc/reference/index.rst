.. _api:

API reference
=============

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2024


The Tumult Analytics API reference is split in
two
major sections.

* :ref:`Specifying privacy guarantees<privacy-guarantees>` introduces the
  :class:`~tmlt.analytics.Session`, the fundamental abstraction used to enforce
  differential privacy guarantees on sensitive data. It also lists classes used to
  define privacy budgets and protected changes.
* :ref:`Building queries<queries>` introduces the classes and methods used to
  define statistical queries to evaluate with differential privacy.


Two further pages list :ref:`configuration options<config>` and :ref:`utility
functions<utils>`.


Full table of contents
----------------------

.. toctree::
   :maxdepth: 2

   privacy-guarantees
   queries
   config
   utils
