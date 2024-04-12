"""Tumult Analytics is a differentially private analytics library from `Tumult Labs`_.

.. _Tumult Labs: https://tmlt.io

The library is broken up into a number of modules that provide different
functionality. These modules fall into a few broad categories, described below.

**For specifying privacy guarantees**:

* :mod:`~.session` defines the :class:`~.session.Session`, an interactive
  interface for making differentially private queries.

* :mod:`~.privacy_budget` contains types for representing privacy budgets.

* :mod:`~.protected_change` contains types for expressing what changes in input
  tables are protected by differential privacy.

**For defining queries**:

* :mod:`~.query_builder` provides an interface for constructing differentially
  private queries from basic query operations.

* :mod:`~.keyset` defines :class:`~.keyset.KeySet`, a type for specifying the
  keys used in group-by operations.

* :mod:`~.constraints` contains types for representing constraints used when
  evaluating queries on data with the :class:`~.protected_change.AddRowsWithID`
  protected change.

* :mod:`~.truncation_strategy` and :mod:`~.binning_spec` provide types that are used
  to define certain types of queries.

"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

# These gets automatically replaced by the version number during the release process
# by poetry-dynamic-versioning.
__version__ = "0.0.0"
__version_tuple__ = (0, 0, 0)

from typing import List

from tmlt.core.utils.configuration import check_java11

__all__: List[str] = []

try:
    # Addresses https://nvd.nist.gov/vuln/detail/CVE-2023-47248 for Python 3.7
    # Python 3.8+ resolve this by using PyArrow >=14.0.1, so it may not be available
    import pyarrow_hotfix
except ImportError:
    pass

check_java11()
