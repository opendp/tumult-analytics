.. _Installation instructions:

Installation instructions
=========================

This guide will help you set up Tumult Analytics on your local machine.

Prerequisites
^^^^^^^^^^^^^

Tumult Analytics is built in `Python <https://www.python.org/>`__, so a Python installation is required to use it.
Only Python 3.7 is officially supported at present, though newer versions likely work.

..
   TODO: Update this once we support PySpark >3.1.0, as the requirements change.

Because Tumult Analytics uses PySpark for computation, it also `requires Java 8 or 11 <https://spark.apache.org/docs/3.0.0/index.html#downloading>`__.

Tumult Analytics uses `python-flint <https://github.com/fredrik-johansson/python-flint/>`__
for some mathematical functions, so ``python-flint`` is also required.
Only Tumult's wheels (located at ``https://d3p0voevd56kj6.cloudfront.net``) are
currently supported.

Only the x86_64 processor architecture is officially supported at present.

.. note::
    When using Java 11, Spark requires that Java be passed the ``-Dio.netty.tryReflectionSetAccessible=true`` option.
    This can be achieved by adding that option to Spark's ``spark.driver.extraJavaOptions`` and ``spark.executor.extraJavaOptions`` configurations.
    An example of how to set these values when initializing Spark is :ref:`included in the first tutorial <Java 11 configuration example>`.
    Not doing this before making calls to Tumult Analytics will result in Spark raising ``java.lang.UnsupportedOperationException: sun.misc.Unsafe or java.nio.DirectByteBuffer.(long, int) not available`` when evaluating queries.


..
   TODO: Link "let us know" to the public issue tracker for Analytics, once it is set up.

Below are instructions for installing these prerequisites on several common platforms.
If none of these apply to you, install Python 3 and Java from your OS package manager, or use the "Manual Installation" section to obtain installers or source distributions and build instructions for your platform.
If you encounter any issues during the installation process, please let us know!

.. tabbed:: Linux (Debian-based)

   Python and ``pip``, Python's package manager, are likely already installed.
   If they are not, install them with:

   .. code-block:: bash

      apt install python3 python3-pip

   Java may already be installed as well.
   If it is not, install the Java Runtime Environment with:

   .. code-block:: bash

      apt install openjdk-8-jre-headless

   Then install python-flint with

   .. code-block:: bash

      pip install -i https://d3p0voevd56kj6.cloudfront.net python-flint

.. tabbed:: Linux (Red Hat-based)

   Python and ``pip``, Python's package manager, may already be installed.
   On some releases, Python 2 may be installed by default, but not Python 3.
   To install Python 3, run:

   .. code-block:: bash

      yum install python3 python3-pip

   To install Java, run:

   .. code-block:: bash

      yum install java-1.8.0-openjdk-headless

   Note that despite the package name, this will install Java 8.

   Then install python-flint with

   .. code-block:: bash

      pip install -i https://d3p0voevd56kj6.cloudfront.net python-flint


.. tabbed:: macOS

    The below instructions assume the use of `Homebrew <https://brew.sh/>`__ for managing packages.
    If you do not wish to use Homebrew for this, use the "Manual Installation" instructions instead.

    If you do not already have Homebrew, it can be installed with:

    .. code-block:: bash

       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    Python may be installed with:

    .. code-block:: bash

       brew install python@3.7

    And Java may be installed with:

    .. code-block:: bash

       brew install openjdk@8

   Then install python-flint with

   .. code-block:: bash

      pip3 install -i https://d3p0voevd56kj6.cloudfront.net python-flint


.. tabbed:: Manual installation

    If Python is not available through your system's package manager, Python installers and source tarballs are available on the `Python downloads page <https://www.python.org/downloads/>`__.

    `Amazon Corretto <https://aws.amazon.com/corretto/>`__ and `Adoptium <https://adoptium.net>`__ (formerly AdoptOpenJDK) both provide Java distributions based on OpenJDK for a variety of platforms.
    Their installation instructions are available `here <https://docs.aws.amazon.com/corretto/latest/corretto-17-ug/what-is-corretto-17.html>`__ and `here <https://adoptium.net/installation.html>`__, respectively, so pick one and get it set up before proceeding.

    ..
        TODO(#1845): Remove this section once Windows support is added.

    .. attention:: If you are installing on a Windows machine, please install `python-flint <https://fredrikj.net/python-flint/>`__ (see `instructions <https://github.com/fredrik-johansson/python-flint/#installation>`__) before installing Tumult Analytics. 



Installation
^^^^^^^^^^^^

..
   TODO: Update this section once the final distribution mechanism is set up.

.. note::
   In order to install Tumult Analytics, you will need a username and token to download it.
   If you do not currently have these, contact `support@tmlt.io <mailto:support@tmlt.io>`_.

Once the above prerequisites are installed, Tumult Analytics can be installed using ``pip`` with:

.. code-block:: bash

   pip install --extra-index-url https://<token-username>:<token>@gitlab.com/api/v4/projects/17405343/packages/pypi/simple tmlt.analytics

This will automatically install all of its Python dependencies as well.

It is recommended, though not required, to install Tumult Analytics in a `virtual environment <https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments>`__ to minimize interactions with your system Python environment.
