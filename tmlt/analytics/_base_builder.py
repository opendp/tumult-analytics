"""Building blocks of modular builders used by various Analytics objects."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Set

from pyspark.sql import DataFrame
from typeguard import check_type, typechecked

from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._utils import assert_is_identifier
from tmlt.analytics.privacy_budget import PrivacyBudget
from tmlt.analytics.protected_change import AddRowsWithID, ProtectedChange


class BaseBuilder(ABC):
    """A base for various builders of privacy-tracking objects."""

    @abstractmethod
    def build(self) -> Any:
        """Constructs the type that this builder builds."""


class PrivacyBudgetMixin:
    """Adds support for setting the privacy budget for a builder."""

    __budget: Optional[PrivacyBudget] = None

    def __init__(self):
        """Constructor.

        @nodoc
        """
        super().__init__()
        self.__budget = None

    @typechecked
    def with_privacy_budget(self, privacy_budget: PrivacyBudget):
        """Set the privacy budget for the object being built."""
        check_type("privacy_budget", privacy_budget, PrivacyBudget)
        if self.__budget is not None:
            raise ValueError("This builder already has a privacy budget set")
        self.__budget = privacy_budget
        return self

    @property
    def _privacy_budget(self) -> PrivacyBudget:
        if self.__budget is None:
            raise ValueError("This builder must have a privacy budget set")
        return self.__budget


class PrivateDataFrame(NamedTuple):
    """A private dataframe and its protected change."""

    dataframe: DataFrame
    protected_change: ProtectedChange


class DataFrameMixin:
    """Adds private and public dataframe support to a builder."""

    __private_dataframes: Dict[str, PrivateDataFrame]
    __public_dataframes: Dict[str, DataFrame]
    __id_spaces: Set[str]

    def __init__(self):
        """Constructor.

        @nodoc
        """
        super().__init__()
        self.__private_dataframes = {}
        self.__public_dataframes = {}
        self.__id_spaces = set()

    @typechecked
    def with_private_dataframe(
        self,
        source_id: str,
        dataframe: DataFrame,
        protected_change: ProtectedChange,
    ):
        """Adds a Spark DataFrame as a private source.

        Not all Spark column types are supported in private sources; see
        :data:`tmlt.analytics.session.SUPPORTED_SPARK_TYPES` for information about
        which types are supported.

        Args:
            source_id: Source id for the private source dataframe.
            dataframe: Private source dataframe to perform queries on,
                corresponding to the ``source_id``.
            protected_change: A
                :class:`~tmlt.analytics.protected_change.ProtectedChange`
                specifying what changes to the input data should be protected.
        """
        assert_is_identifier(source_id)
        if (
            source_id in self.__private_dataframes
            or source_id in self.__public_dataframes
        ):
            raise ValueError(f"Table '{source_id}' already exists")

        dataframe = coerce_spark_schema_or_fail(dataframe)
        self.__private_dataframes[source_id] = PrivateDataFrame(
            dataframe, protected_change
        )
        return self

    @typechecked
    def with_public_dataframe(self, source_id: str, dataframe: DataFrame):
        """Adds a public dataframe."""
        assert_is_identifier(source_id)
        if (
            source_id in self.__private_dataframes
            or source_id in self.__public_dataframes
        ):
            raise ValueError(f"Table '{source_id}' already exists")

        dataframe = coerce_spark_schema_or_fail(dataframe)
        self.__public_dataframes[source_id] = dataframe
        return self

    @typechecked
    def with_id_space(self, id_space: str):
        """Adds an identifier space.

        This defines a space of identifiers that map 1-to-1 to the identifiers
        being protected by a table with the :class:`~.AddRowsWithID` protected
        change. Any table with such a protected change must be a member of some
        identifier space.
        """
        assert_is_identifier(id_space)
        if id_space in self.__id_spaces:
            raise ValueError(f"ID space '{id_space}' already exists")
        self.__id_spaces.add(id_space)
        return self

    def _add_id_space_if_one_private_df(self):
        """If there's only one private dataframe, add its ID space.

        This only has any effect if:
        - there is only one private DataFrame, and
        - that private DataFrame uses the :class:`~.AddRowsWithID` protected change, and
        - this builder does not already have the associated ID space.
        """
        if len(self._private_dataframes) != 1:
            return self
        only_protected_change = list(self._private_dataframes.values())[
            0
        ].protected_change
        if not isinstance(only_protected_change, AddRowsWithID):
            return self
        id_space = only_protected_change.id_space
        if id_space in self._id_spaces:
            return self
        self.with_id_space(id_space)
        return self

    @property
    def _private_dataframes(self) -> Dict[str, PrivateDataFrame]:
        return dict(self.__private_dataframes)

    @property
    def _public_dataframes(self) -> Dict[str, DataFrame]:
        return dict(self.__public_dataframes)

    @property
    def _id_spaces(self) -> Set[str]:
        return self.__id_spaces


class ParameterMixin:
    """Adds support for setting parameters to a builder."""

    __parameters: Dict[str, Any]

    def __init__(self):
        """Constructor.

        @nodoc
        """
        super().__init__()
        self.__parameters = {}

    @typechecked
    def with_parameter(self, name: str, value: Any):
        """Set the value of a parameter."""
        check_type("name", name, str)
        if name in self.__parameters:
            raise ValueError(f"Parameter '{name}' has already been set")
        self.__parameters[name] = value
        return self

    @property
    def _parameters(self) -> Dict[str, Any]:
        return dict(self.__parameters)
