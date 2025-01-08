"""Defines :class:`~tmlt.analytics.Constraint` types.

Constraints are necessary for most aggregations on tables using the
:class:`~tmlt.analytics.AddRowsWithID`
:class:`~tmlt.analytics.ProtectedChange`.

Illustrated examples using constraints can be found in the
:ref:`Working with privacy IDs` tutorial.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

# Some manual import sorting here so that classes appear in the order we want in
# the docs, as opposed to the order isort puts them in.

from ._base import Constraint
from ._simplify import simplify_constraints

from ._truncation import (  # isort:skip
    MaxRowsPerID,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
)
