"""Base class for KeySet operations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from abc import ABC, abstractmethod
from typing import Literal, Optional, overload

from pyspark.sql import DataFrame

from tmlt.analytics._schema import ColumnDescriptor


class KeySetOp(ABC):
    """Base class for operations used to define KeySets."""

    @abstractmethod
    def columns(self) -> set[str]:
        r"""Get a list of the columns included in the output of this operation.

        The column order of a :class:`KeySetOp` is an implementation detail, and
        should not be considered when deciding whether two :class:`KeySetOp`\ s
        are equivalent.
        """

    @abstractmethod
    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation.

        If this operation is a plan (i.e. ``self.is_plan()`` returns True), this
        method will raise ``AnalyticsInternalError``.
        """

    @abstractmethod
    def dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.

        If this operation is a plan (i.e. ``self.is_plan()`` returns True), this
        method will raise ``AnalyticsInternalError``.
        """

    @abstractmethod
    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""

    @abstractmethod
    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""

    @overload
    def size(self, fast: Literal[True]) -> Optional[int]:
        ...

    @overload
    def size(self, fast: Literal[False]) -> int:
        ...

    # Needed to make mypy happy, as it won't automatically combine the above
    # overloads to understand that fast is a bool.
    #   https://github.com/python/mypy/issues/10194
    @overload
    def size(self, fast: bool) -> Optional[int]:
        ...

    @abstractmethod
    def size(self, fast):
        """Determine the size of the KeySet resulting from this operation."""
