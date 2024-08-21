"""Defines :class:`~tmlt.analytics.constraints.Constraint` types.

Constraints are necessary for most aggregations on tables using the
:class:`~tmlt.analytics.protected_change.AddRowsWithID`
:class:`~tmlt.analytics.protected_change.ProtectedChange`.

Illustrated examples using constraints can be found in the
:ref:`Working with privacy IDs` tutorial.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

# Some manual import sorting here so that classes appear in the order we want in
# the docs, as opposed to the order isort puts them in.

from ._base import Constraint
from ._simplify import simplify_constraints

from ._truncation import (  # isort:skip
    MaxRowsPerID,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
)
