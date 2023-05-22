"""Defines the base :class:`Constraint` class."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023


from abc import ABC, abstractmethod
from typing import Tuple

from tmlt.core.transformations.base import Transformation

from tmlt.analytics._table_reference import TableReference


class Constraint(ABC):
    """A known, enforceable fact about a table.

    Constraints provide information about the contents of a table to help
    produce differentially-private results. For example, a constraint might say
    that each ID in a table corresponds to no more than two rows in that table
    (the :class:`~tmlt.analytics.constraints.MaxRowsPerID`
    constraint). Constraints are applied via the :meth:`QueryBuilder.enforce()
    <tmlt.analytics.query_builder.QueryBuilder.enforce>` method.

    This class is a base class for all constraints, and cannot be used directly.
    """

    @abstractmethod
    def _enforce(
        self, child_transformation: Transformation, child_ref: TableReference
    ) -> Tuple[Transformation, TableReference]:
        raise NotImplementedError()
