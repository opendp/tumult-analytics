.. _Privacy promise:

Tumult Analytics' privacy promise
=================================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2023

This topic guide outlines the "privacy promise" provided by Tumult Analytics,
along with its caveats. This guarantee is based on one of the core abstractions
of Tumult Analytics: :class:`Sessions<tmlt.analytics.session.Session>`.

At a high level, a Session allows you to evaluate queries on private data in a
way that satisfies differential privacy. When creating a Session, private data
must first be loaded into it, along with a
:ref:`privacy budget<Privacy budget fundamentals>`. You can then evaluate
queries on your private data, consuming at most the privacy budget provided at
initialization time.

The privacy promise in more detail
----------------------------------

A :class:`~tmlt.analytics.session.Session` is initialized with:

* one or more private data sources (data you wish to query in a differentially
  private way);
* one or more public data sources (data that does not require privacy
  protection, but may be used as auxiliary input to your computation);
* a privacy definition along with its associated privacy parameters (e.g.
  tutorials use `PureDBBudget`, corresponding to pure differential privacy, and
  Tumult Analytics also supports zero-concentrated differential privacy).

After initialization, the Session guarantees that the answers returned by
calling :meth:`~tmlt.analytics.session.Session.evaluate` to evaluate queries
satisfy the corresponding privacy definition with respect to the private data,
using the specified parameters. For example, a Session initialized with
:code:`PureDPBudget(1)` provides :math:`{\varepsilon}`-differential privacy with
:math:`{\varepsilon}=1`.

.. _privacy-promise#unit-of-protection:

Subtlety 1: unit of protection
------------------------------

In the simplest case, the privacy guarantee from a :class:`~tmlt.analytics.session.Session` prevents an attacker from learning whether *one individual row* was added or removed in each private dataset -- the :class:`~tmlt.analytics.protected_change.AddOneRow` protected change.
If the data of a single individual can span multiple rows in the same private dataset, then this individual is not covered by the privacy promise, only individual rows are.

If you know that a single individual can appear in at most *k* rows in an input dataset, you can load that dataset into a Session using a different value for the ``protected_change`` parameter.
For example, using ``AddMaxRows(k)`` will cause Tumult Analytics to hide the addition or removal of up to *k* rows at once from the corresponding private dataset, providing individual-level protection.
Other possible protected changes are also available, though they are typically only needed for advanced use cases.

Because the privacy of individuals depends on how often they appear in a dataset, you should be careful of what kind of pre-processing is done to the private data before loading it into a Session.
If you start from a DataFrame where each individual appears in a single record, but this property stops being true before the data is loaded into a Session, then you might not get the expected privacy guarantees.

Subtlety 2: covered inputs & outputs
------------------------------------

A Session only provides guarantees on the private datasets, and this guarantee
only covers data returned by ``evaluate`` calls. Use of the private data in any
other way is not protected by the Session.

This means that **you should not directly use private data**; instead, you
should only access it indirectly by executing
:meth:`~tmlt.analytics.session.Session.evaluate` on well-specified queries. In
particular, public sources and parameters like ``groupby`` information or
clamping bounds are not protected. They can reveal private information if the
private data is used directly to determine them.

Subtlety 3: adversarial model
-----------------------------

Tumult Analytics, and in particular the Session interface, is designed to make
it easy to obtain expected differential privacy guarantees, and difficult to
accidentally break these guarantees. However, this library was *not* designed to
defend against actively malicious users. In particular:

#. **Do not inspect the private state of a Session or other objects.** The
   privacy guarantees of a Session only apply to the public API. Inspecting a
   Session's private state and using this information to tailor your analysis
   workflow will break the privacy guarantee.

#. **Do not use** :meth:`~tmlt.analytics.query_builder.QueryBuilder.map` **or** :meth:`~tmlt.analytics.query_builder.QueryBuilder.flat_map` **operations with side-effects.**
   These operations allow you to transform data using arbitrary user-defined
   functions (UDFs). When using map or flatmap, a Session's privacy guarantee
   only holds if the UDFs do not have side-effects with externally-observable
   behaviors. For example, a UDF could be designed to throw an exception if a
   specific record is found in the data. This would reveal information about the
   private data and break the privacy promise.

#. **Do not release side-channel information.** The privacy guarantee only
   applies to the output of calls to
   :meth:`~tmlt.analytics.session.Session.evaluate`. Information such as how
   long a query ran or how much memory it required might reveal private
   information. Do not use this library in an untrusted context where protection
   against such side-channels is important.
