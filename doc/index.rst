Tumult Analytics documentation
==============================

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <installation>
   tutorials/index
   topic-guides/index
   API reference <reference/tmlt/analytics/index>
   additional-resources/index

Tumult Analytics is a Python library for computing aggregate queries on tabular
data using differential privacy.

Tumult Analytics is…

- … *robust*: it is built and maintained by a team of differential privacy
  experts, and runs in production at institutions like the U.S. Census Bureau.
- … *scalable*: it runs on `Spark <http://spark.apache.org>`__, so it can scale
  to very large datasets.
- … *easy to use*: its interface will seem familiar to anyone with prior
  experience with tools like SQL or
  `PySpark <http://spark.apache.org/docs/latest/api/python/>`__.
- … *feature-rich*: it supports a large and ever-growing list of aggregation
  functions, data transformation operators, and privacy definitions.

New users probably want to start with the :ref:`installation instructions`.

.. panels::
   :card: + intro-card text-center
   :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 p-2

   ---
   :img-top: /images/index_tutorials.svg

   **Tutorials**
   ^^^^^^^^^^^^^

   Tutorials are the place where new users can learn the basics of how to use
   the library. No prior knowledge of differential privacy is required!

   .. link-button:: tutorials/index
       :type: ref
       :text:
       :classes: stretched-link

   ---
   :img-top: images/index_topic_guides.svg

   **Topic guides**
   ^^^^^^^^^^^^^^^^

   Topic guides dive deeper into specific aspects of the library, and explain in
   more detail how it works behind the scenes.

   .. link-button:: topic-guides/index
       :type: ref
       :text:
       :classes: stretched-link

   ---
   :img-top: images/index_api.svg

   **API reference**
   ^^^^^^^^^^^^^^^^^

   The API reference contains a detailed description of the packages, classes,
   and methods available in Tumult Analytics. It assumes that you have an
   understanding of the key concepts.

   .. link-button:: reference/tmlt/analytics/index
       :type: ref
       :text:
       :classes: stretched-link

   ---
   :img-top: images/index_more.svg


   **Additional resources**
   ^^^^^^^^^^^^^^^^^^^^^^^^

   Additional resources include the :ref:`changelog <Changelog>`, which
   describes notable changes to the library, :ref:`contact information
   <Contact>`, as well as :ref:`license information <License>`.

Documentation License
^^^^^^^^^^^^^^^^^^^^^
This documentation is licensed under the
Creative Commons Attribution-ShareAlike 4.0 Unported License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
