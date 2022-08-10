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


You may also need to configure pyspark to use the intended Python installation. You can do this by setting the `PYSPARK_PYTHON` environment variable. For example:

    .. code-block:: bash

        export PYSPARK_PYTHON=/usr/bin/python3

Tumult Analytics uses `python-flint <https://github.com/fredrik-johansson/python-flint/>`__
for some mathematical functions, so ``python-flint`` is also required.
Only Tumult's wheels (located at ``https://d3p0voevd56kj6.cloudfront.net``) are
currently supported.

Only the x86_64 processor architecture is officially supported at present. Apple silicon is supported through binary translation with Rosetta 2.

.. note::
    When using Java 11, Spark requires that Java be passed the ``-Dio.netty.tryReflectionSetAccessible=true`` option.
    This can be achieved by adding that option to Spark's ``spark.driver.extraJavaOptions`` and ``spark.executor.extraJavaOptions`` configurations.
    An example of how to set these values when initializing Spark is :ref:`included in the first tutorial <Java 11 configuration example>`.
    Not doing this before making calls to Tumult Analytics will result in Spark raising ``java.lang.UnsupportedOperationException: sun.misc.Unsafe or java.nio.DirectByteBuffer.(long, int) not available`` when evaluating queries.

Below are instructions for installing these prerequisites on several common platforms.
If none of these apply to you, install Python 3 and Java from your OS package manager, or use the "Manual Installation" section to obtain installers or source distributions and build instructions for your platform.
If you encounter any issues during the installation process, please `let us know <https://gitlab.com/tumult-labs/analytics/-/issues>`__!

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

        pip3 install -i https://d3p0voevd56kj6.cloudfront.net python-flint
    

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

        pip3 install -i https://d3p0voevd56kj6.cloudfront.net python-flint


.. tabbed:: macOS (Intel)

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

    Then install python-flint with:

    .. code-block:: bash

        pip3 install -i https://d3p0voevd56kj6.cloudfront.net python-flint


.. tabbed:: macOS (Apple silicon)

    Since Python 3.7 is not supported on the Apple silicon processor architecture, you will need to first install `Rosetta 2 <https://support.apple.com/en-us/HT211861>`__ and the x86_64 version of Homebrew.

    If you do not already have Rosetta 2, it can be installed with:

    .. code-block:: bash

        softwareupdate --install-rosetta

    The x86_64 version of Homebrew can be installed with:

    .. code-block:: bash

        arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

    Now, you can install Python 3.7 with:

    .. code-block:: bash

        arch -x86_64 /usr/local/bin/brew install python@3.7

    And Java may be installed with:

    .. code-block:: bash

        arch -x86_64 /usr/local/bin/brew install openjdk@8

    Then install python-flint with:

    .. code-block:: bash

        pip3 install -i https://d3p0voevd56kj6.cloudfront.net python-flint

.. tabbed:: Windows

    The only supported way to install Tumult Analytics on Windows is using the `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/about>`__.

    Once you install your preferred Linux distribution with WSL, follow the corresponding Linux installation instructions to get Tumult Core setup.


.. note::
    Tumult's wheels for python-flint may require a newer version of pip than is included your Python version. If you encounter an error during installation, try upgrading with:
    
    .. code-block:: bash

        pip3 install --upgrade pip
    
    Then retry installation. 

Installation
^^^^^^^^^^^^

Once the above prerequisites are installed, Tumult Analytics can be installed using ``pip3`` with:

.. code-block:: bash

  pip3 install tmlt.analytics

This will automatically install all of its Python dependencies as well.

It is recommended, though not required, to install Tumult Analytics in a `virtual environment <https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments>`__ to minimize interactions with your system Python environment.
