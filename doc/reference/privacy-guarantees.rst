.. _privacy-guarantees:

Specifying privacy guarantees
=============================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2025

The :class:`~tmlt.analytics.Session` is the main object used to specify formal
privacy guarantees on sensitive data. Users specify privacy guarantees at
Session initialization time, using one
:ref:`protected change<protected-changes>` per sensitive table, and an overall
:ref:`privacy budget<privacy-budgets>`. Together, these define the formal
guarantee that the Session then enforces.

Once the :class:`~tmlt.analytics.Session` is initialized, it then ensures that
all future interactions with it satisfy the specified privacy guarantee. In
particular, queries evaluated using :meth:`~tmlt.analytics.Session.evaluate`
cannot consume more than the specified privacy budget.

A simple introduction to Session initialization and use can be found in the
:ref:`First steps<first-steps>` and
:ref:`Working with privacy budgets<privacy-budget-basics>` tutorials. More
details on the exact privacy promise provided by the
:class:`~tmlt.analytics.Session` can be found in the
:ref:`Privacy promise<privacy-promise>` topic guide.

.. currentmodule:: tmlt.analytics

Session
-------
The :class:`~tmlt.analytics.Session` is the fundamental abstraction used to
enforce formal privacy guarantees on sensitive data.

.. autosummary::
   :toctree: api/
   :nosignatures:

   Session

Initializing the Session
------------------------
Sessions can be initialized using the
:meth:`~tmlt.analytics.Session.from_dataframe` method, or using a
:class:`~tmlt.analytics.Session.Builder`.

.. autosummary::
   :toctree: api/

   Session.from_dataframe
   Session.Builder

.. _protected-changes:

Protected changes
"""""""""""""""""
Each private table in a :class:`~tmlt.analytics.Session` needs a *protected
change*, which describes the maximal change in a table that will be protected by
the privacy guarantees.

.. autosummary::
   :toctree: api/

   ProtectedChange
   AddOneRow
   AddMaxRows
   AddMaxRowsInMaxGroups
   AddRowsWithID

Privacy budgets
"""""""""""""""
Finally, the :class:`~tmlt.analytics.Session` must be initialized with a
*privacy budget*, which quantifies the maximum privacy loss of a differentially
private program. There are different kinds of privacy budgets, depending on
which variant of differential privacy is used for this quantification.

.. _privacy-budgets:

.. autosummary::
   :toctree: api/
  
   PrivacyBudget
   PureDPBudget
   ApproxDPBudget
   RhoZCDPBudget

Inspecting Session state
------------------------
The :class:`~tmlt.analytics.Session` provides multiple properties and methods
allowing users to inspect its state.

.. autosummary::
   :toctree: api/

   Session.private_sources
   Session.public_sources
   Session.public_source_dataframes
   Session.remaining_privacy_budget
   Session.describe

Inspecting specific sources
---------------------------
The schema and properties of each table in a :class:`~tmlt.analytics.Session`
can be inspected using the following methods.

.. autosummary::
   :toctree: api/

   Session.get_schema
   Session.get_column_types
   Session.get_grouping_column
   Session.get_id_column
   Session.get_id_space

Evaluating queries with the Session
-----------------------------------
Once a :class:`~tmlt.analytics.Session` is initialized, users can
:ref:`build queries<queries>` and evaluate them using the
:ref:`relevant Session methods<evaluating-queries>`.
