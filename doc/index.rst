Tumult Analytics documentation
==============================

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <howto-guides/installation>
   tutorials/index
   How-to guides <howto-guides/index>
   topic-guides/index
   API reference <reference/tmlt/analytics/index>
   Release notes <changelog>

Tumult Analytics is a Python library for computing aggregate queries on tabular
data using differential privacy.

Tumult Analytics is…

- … *easy to use*: its interface will seem familiar to anyone with prior
  experience with tools like SQL or
  `PySpark <http://spark.apache.org/docs/latest/api/python/>`__.
- … *feature-rich*: it supports a large and ever-growing list of aggregation
  functions, data transformation operators, and privacy definitions.
- … *robust*: it is built and maintained by a team of differential privacy
  experts, and runs in production at institutions like the U.S. Census Bureau.
- … *scalable*: it runs on `Spark <http://spark.apache.org>`__, so it can scale
  to very large datasets.

For new users, `this Colab notebook <https://colab.research.google.com/drive/18J_UrHAKJf52RMRxi4OOpk59dV9tvKxO#offline=true&sandboxMode=true>`__ demonstrates basic features of the library without requiring a local installation.
To explore further or work on larger datasets, a good starting point is the :ref:`installation instructions <installation instructions>`.

The Tumult Analytics documentation introduces all of the concepts necessary to get started producing differentially private results.
Users who wish to learn more about the fundamentals of differential privacy can consult
`this blog post series <https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html>`__
or `this longer introduction <https://privacytools.seas.harvard.edu/files/privacytools/files/pedagogical-document-dp_0.pdf>`__.

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
   :img-top: images/index_howto_guides.svg

   **How-to guides**
   ^^^^^^^^^^^^^^^^

   How-to guides are step-by-step instructions on how to install and
   troubleshoot the library locally or on a cloud platform.

   .. link-button:: howto-guides/index
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

..
   This Additional Resources section forces "Contact Us", etc to be subsubsections.
   Without it, "Contact Us" (and subsequent headers) become subsections,
   which have huge text.

Additional resources
--------------------

Contact us
^^^^^^^^^^
The best place to ask questions, file feature requests, or give feedback about Tumult Analytics is our `Slack server <https://tmlt.dev/slack>`__.
We also use it for announcements of new releases and feature additions.

Cite us
^^^^^^^

If you use Tumult Analytics for a scientific publication, we would appreciate citations to the published software and/or its whitepaper.
Both citations can be found below; for the software citation, please replace the version with the version you are using.

.. code-block::

    @software{tumultanalyticssoftware,
        author = {Tumult Labs},
        title = {Tumult {{Analytics}}},
        month = dec,
        year = 2022,
        version = {latest},
        url = {https://tmlt.dev}
    }


.. code-block::

    @article{tumultanalyticswhitepaper,
        title={Tumult {{Analytics}}: a robust, easy-to-use, scalable, and expressive framework for differential privacy},
        author={Berghel, Skye and Bohannon, Philip and Desfontaines, Damien and Estes, Charles and Haney, Sam and Hartman, Luke and Hay, Michael and Machanavajjhala, Ashwin and Magerlein, Tom and Miklau, Gerome and Pai, Amritha and Sexton, William and Shrestha, Ruchit},
        journal={arXiv preprint arXiv:2212.04133},
        month = dec,
        year={2022}
    }

License
^^^^^^^
The Tumult Analytics source code and documentation are copyright Tumult Labs 2023.

This documentation is licensed under the
Creative Commons Attribution-ShareAlike 4.0 Unported License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

The Tumult Analytics source code is licensed under the Apache License, version 2.0 (`Apache-2.0 <https://gitlab.com/tumult-labs/core/-/blob/dev/LICENSE>`_).

Privacy policy
^^^^^^^^^^^^^^

See our :ref:`privacy policy <privacy-policy>` for how your
information is used.
