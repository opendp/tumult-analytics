"""Interactive query evaluation using a differential privacy framework.

:class:`Session` provides an interface for managing data sources and performing
differentially private queries on them. A simple session with a single private
datasource can be created using :meth:`Session.from_dataframe`, or a more
complex one with multiple datasources can be constructed using
:class:`Session.Builder`. Queries can then be evaluated on the data using
:meth:`Session.evaluate`.

A Session is initialized with a
:class:`~tmlt.analytics.privacy_budget.PrivacyBudget`, and ensures that queries
evaluated on the private data do not consume more than this budget. By default,
a Session enforces this privacy guarantee at the row level: the queries prevent
an attacker from learning whether an individual row has been added or removed in
each of the private tables, provided that the private data is not used elsewhere
in the computation of the queries.

More details on the exact privacy promise provided by :class:`Session` can be
found in the :ref:`Privacy promise topic guide <Privacy promise>`.
"""

# Copyright Tumult Labs 2023
# SPDX-License-Identifier: Apache-2.0

from operator import xor
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast
from warnings import warn

import pandas as pd  # pylint: disable=unused-import
import sympy as sp
from pyspark.sql import SparkSession  # pylint: disable=unused-import
from pyspark.sql import DataFrame
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.interactive_measurements import (
    InactiveAccountantError,
    InsufficientBudgetError,
    PrivacyAccountant,
    PrivacyAccountantState,
    SequentialComposition,
)
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import AddRemoveKeys as AddRemoveKeysMetric
from tmlt.core.metrics import (
    DictMetric,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.dictionary import CreateDictFromValue
from tmlt.core.transformations.identity import Identity
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.configuration import SparkConfigError, check_java11
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.type_utils import assert_never
from typeguard import check_type, typechecked

from tmlt.analytics._catalog import Catalog, PrivateTable, PublicTable
from tmlt.analytics._coerce_spark_schema import (
    SUPPORTED_SPARK_TYPES,
    TYPE_COERCION_MAP,
    coerce_spark_schema_or_fail,
)
from tmlt.analytics._neighboring_relation import (
    AddRemoveKeys,
    AddRemoveRows,
    AddRemoveRowsAcrossGroups,
    Conjunction,
    NeighboringRelation,
)
from tmlt.analytics._neighboring_relation_visitor import NeighboringRelationCoreVisitor
from tmlt.analytics._noise_info import _noise_from_measurement
from tmlt.analytics._privacy_budget_rounding_helper import get_adjusted_budget
from tmlt.analytics._query_expr_compiler import QueryExprCompiler
from tmlt.analytics._schema import (
    Schema,
    spark_dataframe_domain_to_analytics_columns,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics._table_identifier import Identifier, NamedTable, TableCollection
from tmlt.analytics._table_reference import (
    TableReference,
    find_named_tables,
    find_reference,
    lookup_domain,
    lookup_metric,
)
from tmlt.analytics._transformation_utils import (
    delete_table,
    get_table_from_ref,
    persist_table,
    rename_table,
    unpersist_table,
)
from tmlt.analytics._type_checking import is_exact_number_tuple
from tmlt.analytics.constraints import Constraint, MaxGroupsPerID
from tmlt.analytics.privacy_budget import (
    ApproxDPBudget,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
)
from tmlt.analytics.protected_change import (
    AddMaxRows,
    AddMaxRowsInMaxGroups,
    AddOneRow,
    AddRowsWithID,
    ProtectedChange,
)
from tmlt.analytics.query_builder import ColumnType, GroupedQueryBuilder, QueryBuilder
from tmlt.analytics.query_expr import QueryExpr

__all__ = ["Session", "SUPPORTED_SPARK_TYPES", "TYPE_COERCION_MAP"]


class _PrivateSourceTuple(NamedTuple):
    """Named tuple of private Dataframe, domain and protected change."""

    dataframe: DataFrame
    """Private DataFrame."""

    protected_change: ProtectedChange
    """Protected change for this private source."""

    domain: SparkDataFrameDomain
    """Domain of private DataFrame."""


def _generate_neighboring_relation(
    sources: Dict[str, _PrivateSourceTuple]
) -> Conjunction:
    """Convert a collection of private source tuples into a neighboring relation."""
    relations: List[NeighboringRelation] = []
    # this is used only for AddRemoveKeys.
    protected_ids_dict: Dict[str, Dict[str, str]] = {}

    for name, (_, protected_change, _) in sources.items():
        if isinstance(protected_change, AddMaxRows):
            relations.append(AddRemoveRows(name, protected_change.max_rows))
        elif isinstance(protected_change, AddMaxRowsInMaxGroups):
            relations.append(
                AddRemoveRowsAcrossGroups(
                    name,
                    protected_change.grouping_column,
                    max_groups=protected_change.max_groups,
                    per_group=protected_change.max_rows_per_group,
                )
            )
        elif isinstance(protected_change, AddRowsWithID):
            if protected_ids_dict.get(protected_change.id_space) is None:
                protected_ids_dict[protected_change.id_space] = {}
            protected_ids_dict[protected_change.id_space][
                name
            ] = protected_change.id_column
        else:
            raise ValueError(
                f"Unsupported ProtectedChange type: {type(protected_change)}"
            )
    for identifier, table_to_key_column in protected_ids_dict.items():
        relations.append(AddRemoveKeys(identifier, table_to_key_column))
    return Conjunction(relations)


# TODO(2476): Uncomment this once we allow consuming delta, and remove the part
# that only reports epsilon (immediately after the commented section).
def _format_insufficient_budget_msg(
    requested_budget: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
    remaining_budget: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
    privacy_budget: PrivacyBudget,
) -> str:
    """Format message for InsufficientBudgetError."""
    output = ""

    if isinstance(privacy_budget, ApproxDPBudget):
        if is_exact_number_tuple(requested_budget) and is_exact_number_tuple(
            remaining_budget
        ):
            assert isinstance(requested_budget, tuple)
            assert isinstance(remaining_budget, tuple)
            remaining_epsilon = remaining_budget[0].to_float(round_up=True)
            requested_epsilon = requested_budget[0].to_float(round_up=True)
            #   requested_delta = requested_budget[1].to_float(round_up=True)
            #   remaining_delta = remaining_budget[1].to_float(round_up=True)
            #   output += f"\nRequested: ε={requested_epsilon:.3f},"
            #   output += f" δ={requested_delta:.3f}"
            #   output += f"\nRemaining: ε={remaining_epsilon:.3f},"
            #   output += f" δ={remaining_delta:.3f}"
            #   output += "\nDifference: "
            #   lacks_epsilon = remaining_epsilon < requested_epsilon
            #   lacks_delta = remaining_delta < requested_delta
            #   if lacks_epsilon and lacks_delta:
            #       eps_diff = abs(remaining_epsilon - requested_epsilon)
            #       delta_diff = abs(remaining_delta - requested_delta)
            #       if eps_diff >= 0.1 and delta_diff >= 0.1:
            #           output += f"ε={eps_diff:.3f}, δ={delta_diff:.3f}"
            #       elif eps_diff < 0.1:
            #           output += f"ε={eps_diff:.3e}, δ={delta_diff:.3f}"
            #       elif delta_diff < 0.1:
            #           output += f"ε={eps_diff:.3f}, δ={delta_diff:.3e}"
            #   elif lacks_epsilon:
            #       eps_diff = abs(remaining_epsilon - requested_epsilon)
            #       if eps_diff >= 0.1:
            #           output += f"ε={eps_diff:.3f}"
            #       else:
            #           output += f"ε={eps_diff:.3e}"
            #   elif lacks_delta:
            #       delta_diff = abs(remaining_delta - requested_delta)
            #       if delta_diff >= 0.1:
            #           output += f"δ={delta_diff:.3f}"
            #       else:
            #           output += f"δ={delta_diff:.3e}"
            approx_diff = abs(remaining_epsilon - requested_epsilon)
            output += f"\nRequested: ε={requested_epsilon:.3f}"
            output += f"\nRemaining: ε={remaining_epsilon:.3f}"
            if approx_diff >= 0.1:
                output += f"\nDifference: ε={approx_diff:.3f}"
            else:
                output += f"\nDifference: ε={approx_diff:.3e}"
        else:
            raise AssertionError(
                "Unable to convert privacy budget of type"
                f" {type(privacy_budget)} to float or floats. This is"
                " probably a bug; please let us know about it so we can fix it!"
            )
    elif isinstance(privacy_budget, (PureDPBudget, RhoZCDPBudget)):
        assert isinstance(requested_budget, ExactNumber)
        assert isinstance(remaining_budget, ExactNumber)
        if isinstance(privacy_budget, PureDPBudget):
            remaining_epsilon = remaining_budget.to_float(round_up=True)
            requested_epsilon = requested_budget.to_float(round_up=True)
            approx_diff = abs(remaining_epsilon - requested_epsilon)
            output += f"\nRequested: ε={requested_epsilon:.3f}"
            output += f"\nRemaining: ε={remaining_epsilon:.3f}"
            if approx_diff >= 0.1:
                output += f"\nDifference: ε={approx_diff:.3f}"
            else:
                output += f"\nDifference: ε={approx_diff:.3e}"
        elif isinstance(privacy_budget, RhoZCDPBudget):
            remaining_rho = remaining_budget.to_float(round_up=True)
            requested_rho = requested_budget.to_float(round_up=True)
            approx_diff = abs(remaining_rho - requested_rho)
            output += f"\nRequested: 𝝆={requested_rho:.3f}"
            output += f"\nRemaining: 𝝆={remaining_rho:.3f}"
            if approx_diff >= 0.1:
                output += f"\nDifference: 𝝆={approx_diff:.3f}"
            else:
                output += f"\nDifference: 𝝆={approx_diff:.3e}"
    else:
        raise AssertionError(
            f"Unsupported budget types: {type(requested_budget)},"
            f" {type(remaining_budget)}. This is probably a bug, please let us know"
            " about it so we can fix it!"
        )
    return output


class Session:
    """Allows differentially private query evaluation on sensitive data.

    Sessions should not be directly constructed. Instead, they should be created
    using :meth:`from_dataframe` or with a :class:`Builder`.
    """

    class Builder:
        """Builder for :class:`Session`."""

        def __init__(self):
            """Constructor."""
            self._privacy_budget: Optional[PrivacyBudget] = None
            self._private_sources: Dict[str, _PrivateSourceTuple] = {}
            self._public_sources: Dict[str, DataFrame] = {}
            self._id_spaces: List[str] = []

        def build(self) -> "Session":
            """Builds Session with specified configuration."""
            if self._privacy_budget is None:
                raise ValueError("Privacy budget must be specified.")
            if not self._private_sources:
                raise ValueError("At least one private source must be provided.")
            neighboring_relation = _generate_neighboring_relation(self._private_sources)
            tables = {
                source_id: source_tuple.dataframe
                for source_id, source_tuple in self._private_sources.items()
            }
            sess = (
                Session._from_neighboring_relation(  # pylint: disable=protected-access
                    self._privacy_budget, tables, neighboring_relation
                )
            )
            # check list of ARK identifiers agains session's ID spaces
            assert isinstance(neighboring_relation, Conjunction)
            for child in neighboring_relation.children:
                if isinstance(child, AddRemoveKeys):
                    if child.id_space not in self._id_spaces:
                        raise ValueError(
                            "An AddRowsWithID protected change was specified without "
                            "an associated identifier space for the session.\n"
                            f"AddRowsWithID identifier provided: {child.id_space}\n"
                            f"Identifier spaces for the session: {self._id_spaces}"
                        )
            # add public sources
            for source_id, dataframe in self._public_sources.items():
                sess.add_public_dataframe(source_id, dataframe)

            return sess

        def with_privacy_budget(
            self, privacy_budget: PrivacyBudget
        ) -> "Session.Builder":
            """Sets the privacy budget for the Session to be built.

            Args:
                privacy_budget: Privacy Budget to be allocated to Session.
            """
            if self._privacy_budget is not None:
                raise ValueError("This Builder already has a privacy budget")
            self._privacy_budget = privacy_budget
            return self

        def with_private_dataframe(
            self,
            source_id: str,
            dataframe: DataFrame,
            stability: Optional[Union[int, float]] = None,
            grouping_column: Optional[str] = None,
            protected_change: Optional[ProtectedChange] = None,
        ) -> "Session.Builder":
            """Adds a Spark DataFrame as a private source.

            Not all Spark column types are supported in private sources; see
            :data:`SUPPORTED_SPARK_TYPES` for information about which types are
            supported.

            Args:
                source_id: Source id for the private source dataframe.
                dataframe: Private source dataframe to perform queries on,
                    corresponding to the ``source_id``.
                stability: Deprecated: use ``protected_change`` instead, see
                    :ref:`changelog<changelog#protected-change>`.
                grouping_column: Deprecated: use ``protected_change`` instead, see
                    :ref:`changelog<changelog#protected-change>`.
                protected_change: A
                    :class:`~tmlt.analytics.protected_change.ProtectedChange`
                    specifying what changes to the input data the resulting
                    :class:`Session` should protect.
            """
            _assert_is_identifier(source_id)
            if source_id in self._private_sources or source_id in self._public_sources:
                raise ValueError(f"Duplicate source id: '{source_id}'")

            dataframe = coerce_spark_schema_or_fail(dataframe)
            domain = SparkDataFrameDomain.from_spark_schema(dataframe.schema)

            if protected_change is not None:
                if stability is not None:
                    raise ValueError(
                        "stability must not be specified when using protected_change."
                    )
                if grouping_column is not None:
                    raise ValueError(
                        "grouping_column must not be specified when using"
                        " protected_change."
                    )
                self._private_sources[source_id] = _PrivateSourceTuple(
                    dataframe, protected_change, domain
                )
                return self

            if stability is None:
                warn(
                    (
                        "Using a default for protected_change is deprecated. Future"
                        " code should explicitly specify protected_change=AddOneRow()"
                    ),
                    DeprecationWarning,
                )
                if grouping_column is None:
                    protected_change = AddOneRow()
                else:
                    warn(
                        (
                            "Providing a grouping_column parameter instead of a"
                            " protected_change parameter is deprecated"
                        ),
                        DeprecationWarning,
                    )
                    protected_change = AddMaxRowsInMaxGroups(grouping_column, 1, 1)
                    grouping_column = None
            else:
                warn(
                    "Providing a stability instead of a protected_change is deprecated",
                    DeprecationWarning,
                )
                if stability < 1:
                    raise ValueError("Stability must be a positive integer.")

                if grouping_column is None:
                    if not isinstance(stability, int):
                        raise ValueError(
                            "stability must be an integer when no grouping column is"
                            " specified"
                        )
                    protected_change = AddMaxRows(stability)
                else:
                    warn(
                        (
                            "Providing a grouping_column parameter instead of a"
                            " protected_change parameter is deprecated"
                        ),
                        DeprecationWarning,
                    )
                    if not isinstance(stability, (int, float)):
                        raise ValueError("stability must be a numeric value")
                    protected_change = AddMaxRowsInMaxGroups(
                        grouping_column, max_groups=1, max_rows_per_group=stability
                    )
                    grouping_column = None

            self._private_sources[source_id] = _PrivateSourceTuple(
                dataframe, protected_change, domain
            )
            return self

        def with_public_dataframe(
            self, source_id: str, dataframe: DataFrame
        ) -> "Session.Builder":
            """Adds a Spark DataFrame as a public source.

            Not all Spark column types are supported in public sources; see
            :data:`SUPPORTED_SPARK_TYPES` for information about which types are
            supported.

            Args:
                source_id: Source id for the public data source.
                dataframe: Public DataFrame corresponding to the source id.
            """
            _assert_is_identifier(source_id)
            if source_id in self._private_sources or source_id in self._public_sources:
                raise ValueError(f"Duplicate source id: '{source_id}'")
            dataframe = coerce_spark_schema_or_fail(dataframe)
            self._public_sources[source_id] = dataframe
            return self

        def with_id_space(self, id_space: str) -> "Session.Builder":
            """Creates an identifier space for the session.

            This defines a space of identifiers that map 1-to-1 to the
            identifiers being protected by a table with the
            :class:`~tmlt.analytics.protected_change.AddRowsWithID` protected
            change. Any table with such a protected change must be a member of
            some identifier space.

            Args:
                id_space: The name of the identifier space.
            """
            _assert_is_identifier(id_space)
            if id_space in self._id_spaces:
                raise ValueError(
                    f"This Builder already has an ID space of the name: {id_space}."
                )
            self._id_spaces.append(id_space)
            return self

    def __init__(
        self,
        accountant: PrivacyAccountant,
        public_sources: Dict[str, DataFrame],
        compiler: Optional[QueryExprCompiler] = None,
    ) -> None:
        """Initializes a DP session from a queryable.

        This constructor is not intended to be used directly. Use
        :class:`Session.Builder` or ``from_`` constructors instead.
        """
        # pylint: disable=pointless-string-statement
        """
        Args documented for internal use.
            accountant: A PrivacyAccountant.
            public_sources: The public data for the queries.
                Provided as a dictionary {source_id: dataframe}
            compiler: Compiles queries into Measurements,
                which the queryable uses for evaluation.
        """
        # ensure the session is created with java 11
        try:
            check_java11()
        except SparkConfigError as exc:
            raise RuntimeError(
                """It looks like the configuration of your Spark session is
             incompatible with Tumult Analytics. When running Spark on Java 11 or
             higher, you need to set up your Spark session with specific configuration
             options *before* you start Spark. Analytics automatically sets these
             options if you import Analytics before you build your session. For
             troubleshooting information, see our Spark topic guide:
             https://docs.tmlt.dev/analytics/latest/topic-guides/spark.html """
            ) from exc

        check_type("accountant", accountant, PrivacyAccountant)
        check_type("public_sources", public_sources, Dict[str, DataFrame])
        check_type("compiler", compiler, Optional[QueryExprCompiler])

        self._accountant = accountant

        if not isinstance(self._accountant.output_measure, (PureDP, ApproxDP, RhoZCDP)):
            raise ValueError("Accountant is not using PureDP, ApproxDP, or RhoZCDP.")
        if not isinstance(self._accountant.input_metric, DictMetric):
            raise ValueError("The input metric to a session must be a DictMetric.")
        if not isinstance(self._accountant.input_domain, DictDomain):
            raise ValueError("The input domain to a session must be a DictDomain.")
        self._public_sources = public_sources
        if compiler is None:
            compiler = QueryExprCompiler(output_measure=self._accountant.output_measure)
        if self._accountant.output_measure != compiler.output_measure:
            raise ValueError(
                "PrivacyAccountant's output measure is"
                f" {self._accountant.output_measure}, but compiler output measure is"
                f" {compiler.output_measure}."
            )
        self._compiler = compiler
        self._table_constraints: Dict[Identifier, List[Constraint]] = {
            NamedTable(t): [] for t in self.private_sources
        }

    # pylint: disable=line-too-long
    @classmethod
    @typechecked
    def from_dataframe(
        cls,
        privacy_budget: PrivacyBudget,
        source_id: str,
        dataframe: DataFrame,
        stability: Optional[Union[int, float]] = None,
        grouping_column: Optional[str] = None,
        protected_change: Optional[ProtectedChange] = None,
    ) -> "Session":
        """Initializes a DP session from a Spark dataframe.

        Only one private data source is supported with this initialization
        method; if you need multiple data sources, use
        :class:`~tmlt.analytics.session.Session.Builder`.

        Not all Spark column types are supported in private sources; see
        :data:`SUPPORTED_SPARK_TYPES` for information about which types are
        supported.

        ..
            >>> # Set up data
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 1, 0], ["1", 0, 1], ["1", 2, 1]], columns=["A", "B", "X"]
            ...     )
            ... )

        Example:
            >>> spark_data.toPandas()
               A  B  X
            0  0  1  0
            1  1  0  1
            2  1  2  1
            >>> # Declare budget for the session.
            >>> session_budget = PureDPBudget(1)
            >>> # Set up Session
            >>> sess = Session.from_dataframe(
            ...     privacy_budget=session_budget,
            ...     source_id="my_private_data",
            ...     dataframe=spark_data,
            ...     protected_change=AddOneRow(),
            ... )
            >>> sess.private_sources
            ['my_private_data']
            >>> sess.get_schema("my_private_data").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'X': 'INTEGER'}

        Args:
            privacy_budget: The total privacy budget allocated to this session.
            source_id: The source id for the private source dataframe.
            dataframe: The private source dataframe to perform queries on,
                corresponding to the `source_id`.
            stability: Deprecated: use ``protected_change`` instead, see
                :ref:`changelog<changelog#protected-change>`
            grouping_column: Deprecated: use ``protected_change`` instead, see
                :ref:`changelog<changelog#protected-change>`
            protected_change: A
                :class:`~tmlt.analytics.protected_change.ProtectedChange`
                specifying what changes to the input data the resulting
                :class:`Session` should protect.
        """
        # pylint: enable=line-too-long
        session_builder = (
            Session.Builder()
            .with_privacy_budget(privacy_budget=privacy_budget)
            .with_private_dataframe(
                source_id=source_id,
                dataframe=dataframe,
                stability=stability,
                grouping_column=grouping_column,
                protected_change=protected_change,
            )
        )
        if isinstance(protected_change, AddRowsWithID):
            session_builder.with_id_space(protected_change.id_space)
        return session_builder.build()

    @classmethod
    @typechecked
    def _from_neighboring_relation(
        cls,
        privacy_budget: PrivacyBudget,
        private_sources: Dict[str, DataFrame],
        relation: NeighboringRelation,
    ) -> "Session":
        """Initializes a DP session using the provided :class:`NeighboringRelation`.

        Args:
            privacy_budget: The total privacy budget allocated to this session.
            private_sources: The private data to be used in the session.
                Provided as a dictionary {source_id: Dataframe}.
            relation: the :class:`NeighboringRelation` to be used in the session.
        """
        # pylint: disable=protected-access
        output_measure: Union[PureDP, ApproxDP, RhoZCDP]
        sympy_budget: sp.Expr
        if isinstance(privacy_budget, PureDPBudget):
            output_measure = PureDP()
            sympy_budget = privacy_budget._epsilon.expr
        elif isinstance(privacy_budget, ApproxDPBudget):
            output_measure = ApproxDP()
            if privacy_budget.is_infinite:
                warn(
                    (
                        "The use of ApproxDP is not yet fully supported. Because you"
                        " selected an infinite ApproxDP budget, your session will be"
                        " initialized with PureDP using an infinite epsilon budget."
                    ),
                    UserWarning,
                )
                sympy_budget = (
                    ExactNumber.from_float(float("inf"), round_up=False).expr,
                    0,
                )
            else:
                warn(
                    "The use of ApproxDP is not yet fully supported. Your session"
                    " will be initialized with PureDP using the epsilon provided.",
                    UserWarning,
                )
                sympy_budget = (
                    privacy_budget._epsilon.expr,
                    privacy_budget._delta.expr,
                )
        elif isinstance(privacy_budget, RhoZCDPBudget):
            output_measure = RhoZCDP()
            sympy_budget = privacy_budget._rho.expr
        # pylint: enable=protected-access
        else:
            raise ValueError(
                f"Unsupported PrivacyBudget variant: {type(privacy_budget)}"
            )
        # ensure we have a valid source dict for the NeighboringRelation,
        # raising exception if not.
        relation.validate_input(private_sources)

        # Wrap relation in a Conjunction so that output is appropriate for
        # PrivacyAccountant
        domain, metric, distance, dataframes = Conjunction(relation).accept(
            NeighboringRelationCoreVisitor(private_sources, output_measure)
        )

        compiler = QueryExprCompiler(output_measure=output_measure)

        measurement = SequentialComposition(
            input_domain=domain,
            input_metric=metric,
            d_in=distance,
            privacy_budget=sympy_budget,
            output_measure=output_measure,
        )
        accountant = PrivacyAccountant.launch(measurement, dataframes)
        return Session(accountant=accountant, public_sources={}, compiler=compiler)

    @property
    def private_sources(self) -> List[str]:
        """Returns the ids of the private sources."""
        table_refs = find_named_tables(self._input_domain)
        return [
            t.identifier.name
            for t in table_refs
            if isinstance(t.identifier, NamedTable)
        ]

    @property
    def public_sources(self) -> List[str]:
        """Returns the ids of the public sources."""
        return list(self._public_sources)

    @property
    def public_source_dataframes(self) -> Dict[str, DataFrame]:
        """Returns a dictionary of public source dataframes."""
        return self._public_sources

    @property
    def remaining_privacy_budget(self) -> PrivacyBudget:
        """Returns the remaining privacy_budget left in the session.

        The type of the budget (e.g., PureDP or rho-zCDP) will be the same as
        the type of the budget the Session was initialized with.
        """
        output_measure = self._accountant.output_measure
        privacy_budget_value = self._accountant.privacy_budget

        # mypy doesn't know that the ApproxDP budget is a tuple and PureDP and RhoZCDP are not
        if output_measure == ApproxDP():
            epsilon_budget_value, delta_budget_value = privacy_budget_value  # type: ignore
            return ApproxDPBudget(epsilon_budget_value, delta_budget_value)
        elif output_measure == PureDP():
            return PureDPBudget(privacy_budget_value)  # type: ignore
        elif output_measure == RhoZCDP():
            return RhoZCDPBudget(privacy_budget_value)  # type: ignore
        raise RuntimeError(
            "Unexpected behavior in remaining_privacy_budget. Please file a bug report."
        )

    @property
    def _input_domain(self) -> DictDomain:
        """Returns the input domain of the underlying queryable."""
        if not isinstance(self._accountant.input_domain, DictDomain):
            raise AssertionError(
                "Session accountant's input domain has an incorrect type. This is "
                "probably a bug; please let us know about it so we can "
                "fix it!"
            )
        return self._accountant.input_domain

    @property
    def _input_metric(self) -> DictMetric:
        """Returns the input metric of the underlying accountant."""
        if not isinstance(self._accountant.input_metric, DictMetric):
            raise AssertionError(
                "Session accountant's input metric has an incorrect type. This is "
                "probably a bug; please let us know about it so we can "
                "fix it!"
            )
        return self._accountant.input_metric

    def describe(
        self,
        obj: Optional[Union[QueryExpr, QueryBuilder, GroupedQueryBuilder, str]] = None,
    ) -> None:
        """Describe a Session, table, or query.

        If ``obj`` is not specified, ``session.describe()`` will describe the
        Session and all of the tables it contains.

        If ``obj`` is a :class:`~tmlt.analytics.query_builder.QueryBuilder` or
        :class:`~tmlt.analytics.query_expr.QueryExpr`, ``session.describe(obj)``
        will describe the table that would result from that query if it were
        applied to the Session.

        If ``obj`` is a string, ``session.describe(obj)`` will describe the table
        with that name. This is a shorthand for
        ``session.describe(QueryBuilder(obj))``.

        ..
            >>> # Set up data
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 1, 0], ["1", 0, 1], ["1", 2, 1]], columns=["A", "B", "X"]
            ...     )
            ... )
            >>> # construct session
            >>> sess = Session.from_dataframe(
            ...     PureDPBudget(1),
            ...     "my_private_data",
            ...     spark_data,
            ...     protected_change=AddOneRow(),
            ... )

        Examples:
            >>> # describe a session, "sess"
            >>> sess.describe() # doctest: +NORMALIZE_WHITESPACE
            The session has a remaining privacy budget of PureDPBudget(epsilon=1).
            The following private tables are available:
            Table 'my_private_data' (no constraints):
                Columns:
                    - 'A'  VARCHAR
                    - 'B'  INTEGER
                    - 'X'  INTEGER
            >>> # describe a query object
            >>> query = QueryBuilder("my_private_data").drop_null_and_nan(["B", "X"])
            >>> sess.describe(query) # doctest: +NORMALIZE_WHITESPACE
            Columns:
                - 'A'  VARCHAR
                - 'B'  INTEGER, not null
                - 'X'  INTEGER, not null
            >>> # describe a table by name
            >>> sess.describe("my_private_data") # doctest: +NORMALIZE_WHITESPACE
            Columns:
                - 'A'  VARCHAR
                - 'B'  INTEGER
                - 'X'  INTEGER

        Args:
            obj: The table or query to be described, or None to describe the
                whole Session.
        """
        if obj is None:
            print(self._describe_self())
        elif isinstance(obj, (QueryExpr, GroupedQueryBuilder)):
            print(self._describe_query_obj(obj))
        elif isinstance(obj, QueryBuilder):
            print(self._describe_query_obj(obj.query_expr))
        elif isinstance(obj, str):
            print(self._describe_query_obj(QueryBuilder(obj).query_expr))
        else:
            assert_never(obj)

    def _describe_self(self) -> str:
        """Describe the current state of this session."""
        out = []
        state = self._accountant.state
        if state == PrivacyAccountantState.ACTIVE:
            # Don't add anything to output if the session is active
            pass
        elif state == PrivacyAccountantState.RETIRED:
            out.append("This session has been stopped, and can no longer be used.")
        elif state == PrivacyAccountantState.WAITING_FOR_CHILDREN:
            out.append(
                "This session is waiting for its children (created with"
                " `partition_and_create`) to finish."
            )
        elif state == PrivacyAccountantState.WAITING_FOR_SIBLING:
            out.append(
                "This session is waiting for its sibling(s) (created with"
                " `partition_and_create`) to finish."
            )
        else:
            raise AssertionError(
                f"Unrecognized accountant state {out}. This is probably a bug; please"
                " let us know about it so we can fix it!"
            )
        budget: PrivacyBudget = self.remaining_privacy_budget
        out.append(f"The session has a remaining privacy budget of {budget}.")
        if len(self._catalog.tables) == 0:
            out.append("The session has no tables available.")
        else:
            public_table_descs = []
            private_table_descs = []
            for name, table in self._catalog.tables.items():
                column_strs = ["\t" + e for e in _describe_schema(table.schema)]
                columns_desc = "\n".join(column_strs)
                if isinstance(table, PublicTable):
                    table_desc = f"Public table '{name}':\n" + columns_desc
                    public_table_descs.append(table_desc)
                elif isinstance(table, PrivateTable):
                    table_desc = f"Table '{name}':\n"
                    table_desc += columns_desc

                    constraints: Optional[
                        List[Constraint]
                    ] = self._table_constraints.get(NamedTable(name))
                    if not constraints:
                        table_desc = (
                            f"Table '{name}' (no constraints):\n" + columns_desc
                        )
                    else:
                        table_desc = (
                            f"Table '{name}':\n" + columns_desc + "\n\tConstraints:\n"
                        )
                        constraints_strs = [f"\t\t- {e}" for e in constraints]
                        table_desc += "\n".join(constraints_strs)

                    private_table_descs.append(table_desc)
                else:
                    raise AssertionError(
                        f"Table {name} has an unrecognized type: {type(table)}. This is"
                        " probably a bug; please let us know about it so we can"
                        " fix it!"
                    )
            if len(private_table_descs) != 0:
                out.append(
                    "The following private tables are available:\n"
                    + "\n".join(private_table_descs)
                )
            if len(public_table_descs) != 0:
                out.append(
                    "The following public tables are available:\n"
                    + "\n".join(public_table_descs)
                )
        return "\n".join(out)

    def _describe_query_obj(
        self, query_obj: Union[QueryExpr, GroupedQueryBuilder]
    ) -> str:
        """Build a description of a query object."""
        if isinstance(query_obj, GroupedQueryBuilder):
            expr = query_obj._query_expr  # pylint: disable=protected-access
        else:
            expr = query_obj
        schema = self._compiler.query_schema(expr, self._catalog)
        schema_desc = _describe_schema(schema)
        constraints: Optional[List[Constraint]] = None
        try:
            constraints = self._compiler.build_transformation(
                query=expr,
                input_domain=self._input_domain,
                input_metric=self._input_metric,
                public_sources=self._public_sources,
                catalog=self._catalog,
                table_constraints=self._table_constraints,
            )[2]
        except NotImplementedError:
            # If the query results in a measurement, this will happen.
            # There are no constraints on measurements, so we can just
            # pass the schema description through.
            pass
        description = "\n".join(schema_desc)
        if constraints:
            description += "\n\tConstraints:\n"
            constraints_strs = [f"\t\t- {e}" for e in constraints]
            description += "\n".join(constraints_strs)
        if isinstance(query_obj, GroupedQueryBuilder):
            ks_df = (
                query_obj._groupby_keys.dataframe()  # pylint: disable=protected-access
            )
            if len(ks_df.columns) > 0:
                description += "\nGrouped on columns "
                col_strs = [f"'{col}'" for col in ks_df.columns]
                description += ", ".join(col_strs)
                description += f" ({ks_df.count()} groups)"
        return description

    @typechecked
    def get_schema(self, source_id: str) -> Schema:
        """Returns the schema for any data source.

        This includes information on whether the columns are nullable.

        Args:
            source_id: The ID for the data source whose column types
                are being retrieved.
        """
        ref = find_reference(source_id, self._input_domain)
        id_space: Optional[str] = None
        if source_id in self.private_sources:
            id_space = self.get_id_space(source_id)
        if ref is not None:
            domain = lookup_domain(self._input_domain, ref)
            return Schema(
                spark_dataframe_domain_to_analytics_columns(domain), id_space=id_space
            )
        else:
            try:
                return Schema(
                    spark_schema_to_analytics_columns(
                        self.public_source_dataframes[source_id].schema
                    ),
                    id_space=id_space,
                )
            except KeyError:
                raise KeyError(
                    f"Table '{source_id}' does not exist. Available tables "
                    f"are: {', '.join(self.private_sources + self.public_sources)}"
                ) from None

    @typechecked
    def get_column_types(self, source_id: str) -> Dict[str, ColumnType]:
        """Returns the column types for any data source.

        This does *not* include information on whether the columns are nullable.
        """
        return {
            key: val.column_type
            for key, val in self.get_schema(source_id).column_descs.items()
        }

    @typechecked
    def get_grouping_column(self, source_id: str) -> Optional[str]:
        """Returns an optional column that must be grouped by in this query.

        When a groupby aggregation is appended to any query on this table, it
        must include this column as a groupby column.

        Args:
            source_id: The ID for the data source whose grouping column
                is being retrieved.
        """
        ref = find_reference(source_id, self._input_domain)
        if ref is None:
            if source_id in self.public_sources:
                raise ValueError(
                    f"Table '{source_id}' is a public table, which cannot have a "
                    "grouping column."
                )
            raise KeyError(
                f"Private table '{source_id}' does not exist. "
                f"Available private tables are: {', '.join(self.private_sources)}"
            )
        metric = lookup_metric(self._input_metric, ref)
        if isinstance(metric, IfGroupedBy) and isinstance(
            metric.inner_metric, (SumOf, RootSumOfSquared)
        ):
            return metric.column
        return None

    @typechecked
    def get_id_column(self, source_id: str) -> Optional[str]:
        """Returns the ID column of a table, if it has one.

        Args:
            source_id: The name of the table whose ID column is being retrieved.
        """
        ref = find_reference(source_id, self._input_domain)
        if ref is None:
            if source_id in self.public_sources:
                raise ValueError(
                    f"Table '{source_id}' is a public table, which cannot have a "
                    "grouping column."
                )
            raise KeyError(
                f"Private table '{source_id}' does not exist. "
                f"Available private tables are: {', '.join(self.private_sources)}"
            )
        metric = lookup_metric(self._input_metric, ref)
        if isinstance(metric, IfGroupedBy) and isinstance(
            metric.inner_metric, SymmetricDifference
        ):
            return metric.column
        return None

    @typechecked
    def get_id_space(self, source_id: str) -> Optional[str]:
        """Returns the ID space of a table, if it has one.

        Args:
            source_id: The name of the table whose ID space is being retrieved.
        """
        # Make sure the table exists
        ref = find_reference(source_id, self._input_domain)
        if ref is None:
            if source_id in self.public_sources:
                raise ValueError(
                    f"Table '{source_id}' is a public table, which cannot have an "
                    "ID space."
                )
            raise KeyError(
                f"Private table '{source_id}' does not exist. "
                f"Available private tables are: {', '.join(self.private_sources)}"
            )
        # Tables not in an ID space will have a parent of ([])
        if ref.parent == TableReference([]):
            return None
        # Otherwise, the parent should be a TableCollection("id_space")
        parent_identifier = ref.parent.identifier
        assert isinstance(parent_identifier, TableCollection), (
            "Expected parent to be a table collection but got"
            f" {parent_identifier} instead. This is probably a bug; please let us know"
            " about it so we can fix it!"
        )
        return parent_identifier.name

    @property
    def _catalog(self) -> Catalog:
        """Returns a Catalog of tables in the Session."""
        catalog = Catalog()
        for table in self.private_sources:
            catalog.add_private_table(
                table,
                self.get_schema(table),
                grouping_column=self.get_grouping_column(table),
                id_column=self.get_id_column(table),
                id_space=self.get_id_space(table),
            )
        for table in self.public_sources:
            catalog.add_public_table(
                table,
                spark_schema_to_analytics_columns(
                    self.public_source_dataframes[table].schema
                ),
            )
        return catalog

    # pylint: disable=line-too-long
    @typechecked
    def add_public_dataframe(self, source_id: str, dataframe: DataFrame):
        """Adds a public data source to the session.

        Not all Spark column types are supported in public sources; see
        :data:`SUPPORTED_SPARK_TYPES` for information about which types are
        supported.

        ..
            >>> # Get data
            >>> spark = SparkSession.builder.getOrCreate()
            >>> private_data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 1, 0], ["1", 0, 1], ["1", 2, 1]], columns=["A", "B", "X"]
            ...     )
            ... )
            >>> public_spark_data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 0], ["0", 1], ["1", 1], ["1", 2]], columns=["A", "C"]
            ...     )
            ... )
            >>> # Set up Session
            >>> sess = Session.from_dataframe(
            ...     privacy_budget=PureDPBudget(1),
            ...     source_id="my_private_data",
            ...     dataframe=private_data,
            ...     protected_change=AddOneRow(),
            ... )

        Example:
            >>> public_spark_data.toPandas()
               A  C
            0  0  0
            1  0  1
            2  1  1
            3  1  2
            >>> # Add public data
            >>> sess.add_public_dataframe(
            ...     source_id="my_public_data", dataframe=public_spark_data
            ... )
            >>> sess.public_sources
            ['my_public_data']
            >>> sess.get_schema('my_public_data').column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'C': 'INTEGER'}

        Args:
            source_id: The name of the public data source.
            dataframe: The public data source corresponding to the ``source_id``.
        """
        # pylint: enable=line-too-long
        _assert_is_identifier(source_id)
        if source_id in self.public_sources or source_id in self.private_sources:
            raise ValueError(f"This session already has a table named '{source_id}'.")
        dataframe = coerce_spark_schema_or_fail(dataframe)
        self._public_sources[source_id] = dataframe

    def _compile_and_get_budget(
        self, query_expr: QueryExpr, privacy_budget: PrivacyBudget
    ) -> Tuple[Measurement, PrivacyBudget]:
        """Pre-processing needed for evaluate() and _noise_info()."""
        check_type("query_expr", query_expr, QueryExpr)
        check_type("privacy_budget", privacy_budget, PrivacyBudget)

        is_approxDP_session = self._accountant.output_measure == ApproxDP()

        # If pureDP session, and approxDP budget, let Core handle the error.
        if is_approxDP_session and isinstance(privacy_budget, PureDPBudget):
            privacy_budget = ApproxDPBudget(privacy_budget.value, 0)

        self._validate_budget_type_matches_session(privacy_budget)
        if privacy_budget in [PureDPBudget(0), ApproxDPBudget(0, 0), RhoZCDPBudget(0)]:
            raise ValueError("You need a non-zero privacy budget to evaluate a query.")

        adjusted_budget = self._process_requested_budget(privacy_budget)

        measurement = self._compiler(
            queries=[query_expr],
            privacy_budget=adjusted_budget,
            stability=self._accountant.d_in,
            input_domain=self._input_domain,
            input_metric=self._input_metric,
            public_sources=self._public_sources,
            catalog=self._catalog,
            table_constraints=self._table_constraints,
        )
        return (measurement, adjusted_budget)

    def _noise_info(
        self, query_expr: QueryExpr, privacy_budget: PrivacyBudget
    ) -> List[Dict[str, Any]]:
        """Get noise information about a query.

        ..
            >>> from tmlt.analytics.keyset import KeySet
            >>> # Get data
            >>> spark = SparkSession.builder.getOrCreate()
            >>> data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 1, 0], ["1", 0, 1], ["1", 2, 1]], columns=["A", "B", "X"]
            ...     )
            ... )
            >>> # Create Session
            >>> sess = Session.from_dataframe(
            ...     privacy_budget=PureDPBudget(1),
            ...     source_id="my_private_data",
            ...     dataframe=data,
            ...     protected_change=AddOneRow(),
            ... )

        Example:
            >>> sess.private_sources
            ['my_private_data']
            >>> sess.get_schema("my_private_data").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'X': 'INTEGER'}
            >>> sess.remaining_privacy_budget
            PureDPBudget(epsilon=1)
            >>> count_query = QueryBuilder("my_private_data").count()
            >>> count_info = sess._noise_info(
            ...     query_expr=count_query,
            ...     privacy_budget=PureDPBudget(0.5),
            ... )
            >>> count_info # doctest: +NORMALIZE_WHITESPACE
            [{'noise_mechanism': <_NoiseMechanism.GEOMETRIC: 2>, 'noise_parameter': 2}]
        """
        measurement, _ = self._compile_and_get_budget(query_expr, privacy_budget)
        return _noise_from_measurement(measurement)

    # pylint: disable=line-too-long
    def evaluate(
        self, query_expr: QueryExpr, privacy_budget: PrivacyBudget
    ) -> DataFrame:
        """Answers a query within the given privacy budget and returns a Spark dataframe.

        The type of privacy budget that you use must match the type your Session was
        initialized with (i.e., you cannot evaluate a query using rho-zCDPBudget if
        the Session was initialized with a PureDPBudget, and vice versa).

        ..
            >>> from tmlt.analytics.keyset import KeySet
            >>> # Get data
            >>> spark = SparkSession.builder.getOrCreate()
            >>> data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 1, 0], ["1", 0, 1], ["1", 2, 1]], columns=["A", "B", "X"]
            ...     )
            ... )
            >>> # Create Session
            >>> sess = Session.from_dataframe(
            ...     privacy_budget=PureDPBudget(1),
            ...     source_id="my_private_data",
            ...     dataframe=data,
            ...     protected_change=AddOneRow(),
            ... )

        Example:
            >>> sess.private_sources
            ['my_private_data']
            >>> sess.get_schema("my_private_data").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'X': 'INTEGER'}
            >>> sess.remaining_privacy_budget
            PureDPBudget(epsilon=1)
            >>> # Evaluate Queries
            >>> filter_query = QueryBuilder("my_private_data").filter("A > 0")
            >>> count_query = filter_query.groupby(KeySet.from_dict({"X": [0, 1]})).count()
            >>> count_answer = sess.evaluate(
            ...     query_expr=count_query,
            ...     privacy_budget=PureDPBudget(0.5),
            ... )
            >>> sum_query = filter_query.sum(column="B", low=0, high=1)
            >>> sum_answer = sess.evaluate(
            ...     query_expr=sum_query,
            ...     privacy_budget=PureDPBudget(0.5),
            ... )
            >>> count_answer # TODO(#798): Seed randomness and change to toPandas()
            DataFrame[X: bigint, count: bigint]
            >>> sum_answer # TODO(#798): Seed randomness and change to toPandas()
            DataFrame[B_sum: bigint]

        Args:
            query_expr: One query expression to answer.
            privacy_budget: The privacy budget used for the query.
        """
        # pylint: enable=line-too-long
        measurement, adjusted_budget = self._compile_and_get_budget(
            query_expr, privacy_budget
        )
        self._activate_accountant()

        # check if type of self._accountant.privacy_budget matches adjusted_budget value
        if xor(
            isinstance(self._accountant.privacy_budget, tuple),
            isinstance(adjusted_budget.value, tuple),
        ):
            raise ValueError(
                "Expected type of adjusted_budget to match type of accountant's privacy"
                f" budget ({type(self._accountant.privacy_budget)}), but instead"
                f" received {type(adjusted_budget.value)}. This is probably a bug;"
                " please let us know about it so we can fix it!"
            )

        try:
            if not measurement.privacy_relation(
                self._accountant.d_in, adjusted_budget.value
            ):
                raise AssertionError(
                    "With these inputs and this privacy budget, similar inputs will"
                    " *not* produce similar outputs. This is probably a bug; please let"
                    " us know about it so we can fix it!"
                )
            try:
                answers = self._accountant.measure(
                    measurement, d_out=adjusted_budget.value
                )
            except InsufficientBudgetError as err:
                msg = _format_insufficient_budget_msg(
                    err.requested_budget, err.remaining_budget, privacy_budget
                )
                raise RuntimeError(
                    "Cannot answer query without exceeding the Session privacy budget."
                    + msg
                ) from err

            if len(answers) != 1:
                raise AssertionError(
                    "Expected exactly one answer, but got "
                    f"{len(answers)} answers instead. This is "
                    "probably a bug; please let us know about it so "
                    "we can fix it!"
                )
            return answers[0]
        except InactiveAccountantError as e:
            raise RuntimeError(
                "This session is no longer active. Either it was manually stopped "
                "with session.stop(), or it was stopped indirectly by the "
                "activity of other sessions. See partition_and_create "
                "for more information."
            ) from e

    # pylint: disable=line-too-long
    @typechecked
    def create_view(
        self, query_expr: Union[QueryExpr, QueryBuilder], source_id: str, cache: bool
    ):
        """Create a new view from a transformation and possibly cache it.

        ..
            >>> # Get data
            >>> spark = SparkSession.builder.getOrCreate()
            >>> private_data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 1, 0], ["1", 0, 1], ["1", 2, 1]], columns=["A", "B", "X"]
            ...     )
            ... )
            >>> public_spark_data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 0], ["0", 1], ["1", 1], ["1", 2]], columns=["A", "C"]
            ...     )
            ... )
            >>> # Create Session
            >>> sess = Session.from_dataframe(
            ...     privacy_budget=PureDPBudget(1),
            ...     source_id="my_private_data",
            ...     dataframe=private_data,
            ...     protected_change=AddOneRow(),
            ... )

        Example:
            >>> sess.private_sources
            ['my_private_data']
            >>> sess.get_schema("my_private_data").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'X': 'INTEGER'}
            >>> public_spark_data.toPandas()
               A  C
            0  0  0
            1  0  1
            2  1  1
            3  1  2
            >>> sess.add_public_dataframe("my_public_data", public_spark_data)
            >>> # Create a view
            >>> join_query = (
            ...     QueryBuilder("my_private_data")
            ...     .join_public("my_public_data")
            ...     .select(["A", "B", "C"])
            ... )
            >>> sess.create_view(
            ...     join_query,
            ...     source_id="private_public_join",
            ...     cache=True
            ... )
            >>> sess.private_sources
            ['private_public_join', 'my_private_data']
            >>> sess.get_schema("private_public_join").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'C': 'INTEGER'}
            >>> # Delete the view
            >>> sess.delete_view("private_public_join")
            >>> sess.private_sources
            ['my_private_data']

        Args:
            query_expr: A query that performs a transformation.
            source_id: The name, or unique identifier, of the view.
            cache: Whether or not to cache the view.
        """
        # pylint: enable=line-too-long
        _assert_is_identifier(source_id)
        self._activate_accountant()
        if source_id in self.private_sources or source_id in self.public_sources:
            raise ValueError(f"Table '{source_id}' already exists.")

        if isinstance(query_expr, QueryBuilder):
            query_expr = query_expr.query_expr

        if not isinstance(query_expr, QueryExpr):
            raise ValueError("query_expr must be of type QueryBuilder or QueryExpr.")
        transformation, ref, constraints = self._compiler.build_transformation(
            query=query_expr,
            input_domain=self._input_domain,
            input_metric=self._input_metric,
            public_sources=self._public_sources,
            catalog=self._catalog,
            table_constraints=self._table_constraints,
        )
        if cache:
            transformation, ref = persist_table(
                base_transformation=transformation, base_ref=ref
            )

        transformation, _ = rename_table(
            base_transformation=transformation,
            base_ref=ref,
            new_table_id=NamedTable(source_id),
        )
        self._accountant.transform_in_place(transformation)
        self._table_constraints[NamedTable(source_id)] = constraints

    def delete_view(self, source_id: str):
        """Deletes a view and decaches it if it was cached.

        Args:
            source_id: The name of the view.
        """
        self._activate_accountant()

        ref = find_reference(source_id, self._input_domain)
        if ref is None:
            raise KeyError(
                f"Private table '{source_id}' does not exist. "
                f"Available tables are: {', '.join(self.private_sources)}"
            )

        domain = lookup_domain(self._input_domain, ref)
        if not isinstance(domain, SparkDataFrameDomain):
            raise RuntimeError(
                "Table domain is not SparkDataFrameDomain. This is probably a bug; "
                "please let us know so we can fix it!"
            )

        unpersist_source: Transformation = Identity(
            domain=self._input_domain, metric=self._input_metric
        )
        # Unpersist does nothing if the DataFrame isn't persisted
        unpersist_source = unpersist_table(
            base_transformation=unpersist_source, base_ref=ref
        )

        transformation = delete_table(
            base_transformation=unpersist_source, base_ref=ref
        )
        self._accountant.transform_in_place(transformation)
        self._table_constraints.pop(ref.identifier, None)

    # pylint: disable=line-too-long
    @typechecked
    def partition_and_create(
        self,
        source_id: str,
        privacy_budget: PrivacyBudget,
        column: Optional[str] = None,
        splits: Optional[Union[Dict[str, str], Dict[str, int]]] = None,
    ) -> Dict[str, "Session"]:
        """Returns new sessions from a partition mapped to split name/``source_id``.

        The type of privacy budget that you use must match the type your Session
        was initialized with (i.e., you cannot use a
        :class:`~tmlt.analytics.privacy_budget.RhoZCDPBudget` to partition your
        Session if the Session was created using a
        :class:`~tmlt.analytics.privacy_budget.PureDPBudget`, and vice versa).

        The sessions returned must be used in the order that they were created.
        Using this session again or calling stop() will stop all partition sessions.

        ..
            >>> # Get data
            >>> spark = SparkSession.builder.getOrCreate()
            >>> data = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         [["0", 1, 0], ["1", 0, 1], ["1", 2, 1]], columns=["A", "B", "X"]
            ...     )
            ... )
            >>> # Create Session
            >>> sess = Session.from_dataframe(
            ...     privacy_budget=PureDPBudget(1),
            ...     source_id="my_private_data",
            ...     dataframe=data,
            ...     protected_change=AddOneRow(),
            ... )
            >>> import doctest
            >>> doctest.ELLIPSIS_MARKER = '...'

        Example:
            This example partitions the session into two sessions, one with A = "0" and
            one with A = "1". Due to parallel composition, each of these sessions are
            given the same budget, while only one count of that budget is deducted from
            session.

            >>> sess.private_sources
            ['my_private_data']
            >>> sess.get_schema("my_private_data").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'X': 'INTEGER'}
            >>> sess.remaining_privacy_budget
            PureDPBudget(epsilon=1)
            >>> # Partition the Session
            >>> new_sessions = sess.partition_and_create(
            ...     "my_private_data",
            ...     privacy_budget=PureDPBudget(0.75),
            ...     column="A",
            ...     splits={"part0":"0", "part1":"1"}
            ... )
            >>> sess.remaining_privacy_budget
            PureDPBudget(epsilon=0.25)
            >>> new_sessions["part0"].private_sources
            ['part0']
            >>> new_sessions["part0"].get_schema("part0").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'X': 'INTEGER'}
            >>> new_sessions["part0"].remaining_privacy_budget
            PureDPBudget(epsilon=0.75)
            >>> new_sessions["part1"].private_sources
            ['part1']
            >>> new_sessions["part1"].get_schema("part1").column_types # doctest: +NORMALIZE_WHITESPACE
            {'A': 'VARCHAR', 'B': 'INTEGER', 'X': 'INTEGER'}
            >>> new_sessions["part1"].remaining_privacy_budget
            PureDPBudget(epsilon=0.75)

            When you are done with a new session, you can use the
            :meth:`~Session.stop` method to allow the next one to become active:

            >>> new_sessions["part0"].stop()
            >>> new_sessions["part1"].private_sources
            ['part1']
            >>> count_query = QueryBuilder("part1").count()
            >>> count_answer = new_sessions["part1"].evaluate(
            ...     count_query,
            ...     PureDPBudget(0.75),
            ... )
            >>> count_answer.toPandas() # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
               count
            0    ...

        Args:
            source_id: The private source to partition.
            privacy_budget: Amount of privacy budget to pass to each new session.
            column: The name of the column partitioning on.
            splits: Mapping of split name to value of partition.
                Split name is ``source_id`` in new session.
        """
        # pylint: enable=line-too-long
        if splits is None:
            raise ValueError(
                "You must provide a dictionary mapping split names (new source_ids) to"
                " values on which to partition"
            )
        # If you remove this if-block, mypy will complain
        if column is None:
            raise AssertionError(
                "column is None, even though either column or attr_name were provided."
                " This is probably a bug; please let us know about it so we can fix it!"
            )

        if not (
            isinstance(self._accountant.privacy_budget, ExactNumber)
            or is_exact_number_tuple(self._accountant.privacy_budget)
        ):
            raise AssertionError(
                "Unable to convert privacy budget of type"
                f" {type(self._accountant.privacy_budget)} to float or floats. This is"
                " probably a bug; please let us know about it so we can fix it!"
            )

        is_approxDP_session = isinstance(self._accountant.output_measure, ApproxDP)
        if is_approxDP_session and isinstance(privacy_budget, PureDPBudget):
            privacy_budget = ApproxDPBudget(privacy_budget.value, 0)

        self._validate_budget_type_matches_session(privacy_budget)
        self._activate_accountant()

        transformation: Transformation = Identity(
            domain=self._input_domain, metric=self._input_metric
        )
        table_ref = find_reference(source_id, self._input_domain)
        if table_ref is None:
            if source_id in self.public_sources:
                raise ValueError(
                    f"Table '{source_id}' is a public table, which cannot have an "
                    "ID space."
                )
            raise KeyError(
                f"Private table '{source_id}' does not exist. "
                f"Available private tables are: {', '.join(self.private_sources)}"
            )

        # Either DictMetric or AddRemoveKeys
        parent_metric = lookup_metric(self._input_metric, table_ref.parent)
        table_has_ids: bool = isinstance(parent_metric, AddRemoveKeysMetric)
        if table_has_ids:
            parent_last_element = table_ref.parent.identifier
            constraints = self._table_constraints.get(NamedTable(source_id))
            if constraints is None:
                raise AssertionError(
                    f"Table '{source_id}' has no constraints. This is probably a"
                    " bug; please let us know about it so we can fix it!"
                )

            constraint = next(
                (
                    c
                    for c in constraints
                    if (isinstance(c, MaxGroupsPerID) and c.grouping_column == column)
                ),
                None,
            )

            if constraint is None:
                raise ValueError(
                    "You must create MaxGroupsPerID constraint before using"
                    " partition_and_create on tables with the AddRowsWithID"
                    " protected change."
                )

            (
                transformation,
                table_ref,
            ) = constraint._enforce(  # pylint: disable=protected-access
                child_transformation=transformation,
                child_ref=table_ref,
                update_metric=True,
                use_l2=isinstance(self._compiler.output_measure, RhoZCDP),
            )
        transformation = get_table_from_ref(transformation, table_ref)
        if not isinstance(
            transformation.output_metric, (IfGroupedBy, SymmetricDifference)
        ):
            raise AssertionError(
                "Transformation has an unrecognized output metric. This is "
                "probably a bug; please let us know about it so we can fix it!"
            )
        transformation_domain = cast(SparkDataFrameDomain, transformation.output_domain)

        try:
            attr_type = transformation_domain.schema[column]
        except KeyError as e:
            raise KeyError(
                f"'{column}' not present in transformed dataframe's columns; "
                "schema of transformed dataframe is "
                f"{spark_dataframe_domain_to_analytics_columns(transformation_domain)}"
            ) from e

        new_sources = []
        # Actual type is Union[List[Tuple[str, ...]], List[Tuple[int, ...]]]
        # but mypy doesn't like that.
        split_vals: List[Tuple[Union[str, int], ...]] = []
        for split_name, split_val in splits.items():
            if not split_name.isidentifier():
                raise ValueError(
                    "The string passed as split name must be a valid Python identifier:"
                    " it can only contain alphanumeric letters (a-z) and (0-9), or"
                    " underscores (_), and it cannot start with a number, or contain"
                    " any spaces."
                )
            if not attr_type.valid_py_value(split_val):
                raise TypeError(
                    f"'{column}' column is of type '{attr_type.data_type}'; "
                    f"'{attr_type.data_type}' column not compatible with splits "
                    f"value type '{type(split_val).__name__}'"
                )
            new_sources.append(split_name)
            split_vals.append((split_val,))

        element_metric: Union[IfGroupedBy, SymmetricDifference]
        if isinstance(transformation.output_metric, SymmetricDifference) or (
            isinstance(transformation.output_metric, IfGroupedBy)
            and column != transformation.output_metric.column
        ):
            element_metric = transformation.output_metric
        elif (
            isinstance(transformation.output_metric, IfGroupedBy)
            and column == transformation.output_metric.column
        ):
            assert isinstance(
                transformation.output_metric.inner_metric,
                (IfGroupedBy, RootSumOfSquared, SumOf),
            )
            assert isinstance(
                transformation.output_metric.inner_metric.inner_metric,
                (IfGroupedBy, SymmetricDifference),
            )
            element_metric = transformation.output_metric.inner_metric.inner_metric
        else:
            raise AssertionError(
                "Transformation has an unrecognized output metric. This is "
                "probably a bug; please let us know about it so  we can fix it!"
            )

        partition_transformation = PartitionByKeys(
            input_domain=transformation_domain,
            input_metric=transformation.output_metric,
            use_l2=isinstance(self._compiler.output_measure, RhoZCDP),
            keys=[column],
            list_values=split_vals,
        )
        chained_partition = transformation | partition_transformation

        adjusted_budget = self._process_requested_budget(privacy_budget)

        try:
            new_accountants = self._accountant.split(
                chained_partition, privacy_budget=adjusted_budget.value
            )
        except InactiveAccountantError as e:
            raise RuntimeError(
                "This session is no longer active. Either it was manually stopped"
                "with session.stop(), or it was stopped indirectly by the "
                "activity of other sessions. See partition_and_create "
                "for more information."
            ) from e
        except InsufficientBudgetError as err:
            msg = _format_insufficient_budget_msg(
                err.requested_budget, err.remaining_budget, privacy_budget
            )
            raise RuntimeError(
                "Cannot perform this partition without exceeding "
                "the Session privacy budget." + msg
            ) from err

        for i, source in enumerate(new_sources):
            if table_has_ids:
                create_dict = CreateDictFromValue(
                    input_domain=transformation_domain,
                    input_metric=element_metric,
                    key=NamedTable(source),
                    use_add_remove_keys=True,
                )
                dict_transformation_wrapper = create_dict | CreateDictFromValue(
                    input_domain=create_dict.output_domain,
                    input_metric=create_dict.output_metric,
                    key=parent_last_element,
                )
            else:
                dict_transformation_wrapper = CreateDictFromValue(
                    input_domain=transformation_domain,
                    input_metric=element_metric,
                    key=NamedTable(source),
                )

            new_accountants[i].queue_transformation(
                transformation=dict_transformation_wrapper
            )

        new_sessions = {}
        for new_accountant, source in zip(new_accountants, new_sources):
            new_sessions[source] = Session(
                new_accountant, self._public_sources, self._compiler
            )
        return new_sessions

    def _process_requested_budget(self, privacy_budget: PrivacyBudget) -> PrivacyBudget:
        """Process the requested budget to accommodate floating point imprecision.

        Args:
            privacy_budget: The requested budget.
        """
        remaining_budget_value = self._accountant.privacy_budget

        if isinstance(privacy_budget, PureDPBudget):
            if not isinstance(remaining_budget_value, ExactNumber):
                raise AssertionError(
                    f"Cannot understand remaining budget of {remaining_budget_value}."
                    " This is probably a bug; please let us know about it so we can"
                    " fix it!"
                )
            return get_adjusted_budget(
                privacy_budget,
                PureDPBudget(remaining_budget_value.to_float(round_up=False)),
            )
        elif isinstance(privacy_budget, ApproxDPBudget):
            if privacy_budget.is_infinite:
                return ApproxDPBudget(float("inf"), 0)
            else:
                if not is_exact_number_tuple(remaining_budget_value):
                    raise AssertionError(
                        "Remaining budget type for ApproxDP must be Tuple[ExactNumber,"
                        " ExactNumber], but instead received"
                        f" {type(remaining_budget_value)}. This is probably a bug;"
                        " please let us know about it so we can fix it!"
                    )
                # mypy doesn't understand that we've already checked that this is a tuple
                remaining_epsilon, remaining_delta = remaining_budget_value  # type: ignore
                return get_adjusted_budget(
                    ApproxDPBudget(*privacy_budget.value),
                    ApproxDPBudget(
                        remaining_epsilon.to_float(round_up=False),
                        remaining_delta.to_float(round_up=False),
                    ),
                )
        elif isinstance(privacy_budget, RhoZCDPBudget):
            if not isinstance(remaining_budget_value, ExactNumber):
                raise AssertionError(
                    f"Cannot understand remaining budget of {remaining_budget_value}."
                    " This is probably a bug; please let us know about it so we can"
                    " fix it!"
                )
            return get_adjusted_budget(
                privacy_budget,
                RhoZCDPBudget(remaining_budget_value.to_float(round_up=False)),
            )
        else:
            raise ValueError(
                f"Unsupported variant of PrivacyBudget. Found {type(privacy_budget)}"
            )

    def _validate_budget_type_matches_session(
        self, privacy_budget: PrivacyBudget
    ) -> None:
        """Ensure that a budget used during evaluate/partition matches the session.

        Args:
            privacy_budget: The requested budget.
        """
        output_measure = self._accountant.output_measure
        matches_puredp = isinstance(output_measure, PureDP) and isinstance(
            privacy_budget, PureDPBudget
        )
        matches_approxdp = isinstance(output_measure, ApproxDP) and isinstance(
            privacy_budget, ApproxDPBudget
        )
        matches_zcdp = isinstance(output_measure, RhoZCDP) and isinstance(
            privacy_budget, RhoZCDPBudget
        )
        if not (matches_puredp or matches_approxdp or matches_zcdp):
            raise ValueError(
                "Your requested privacy budget type must match the type of the"
                " privacy budget your Session was created with."
            )

    def _activate_accountant(self) -> None:
        if self._accountant.state == PrivacyAccountantState.ACTIVE:
            return
        if self._accountant.state == PrivacyAccountantState.RETIRED:
            raise RuntimeError(
                "This session is no longer active, and no new queries can be performed"
            )
        if self._accountant.state == PrivacyAccountantState.WAITING_FOR_SIBLING:
            warn(
                "Activating a session that is waiting for one of its siblings "
                "to finish may cause unexpected behavior."
            )
        if self._accountant.state == PrivacyAccountantState.WAITING_FOR_CHILDREN:
            warn(
                "Activating a session that is waiting for its children "
                "(created with partition_and_create) to finish "
                "may cause unexpected behavior."
            )
        self._accountant.force_activate()

    def stop(self) -> None:
        """Close out this session, allowing other sessions to become active."""
        self._accountant.retire()


def _assert_is_identifier(source_id: str):
    """Checks that the ``source_id`` is a valid Python identifier.

    Args:
        source_id: The name of the dataframe or transformation.
    """
    if not source_id.isidentifier():
        raise ValueError(
            "The string passed as source_id must be a valid Python identifier: it can"
            " only contain alphanumeric letters (a-z) and (0-9), or underscores (_),"
            " and it cannot start with a number, or contain any spaces."
        )


def _describe_schema(schema: Schema) -> List[str]:
    """Get a list of strings to print that describe columns of a schema.

    This is a list so that it's easy to append tabs to each line.
    """
    description = ["Columns:"]
    # We actually care about the maximum length of the column name
    # *as enclosed in quotes*,
    # so we add 2 to account for the opening and closing quotation marks
    column_length = (
        max(len(column_name) for column_name in schema.column_descs.keys()) + 2
    )
    for column_name, cd in schema.column_descs.items():
        quoted_column_name = f"'{column_name}'"
        column_str = f"\t- {quoted_column_name:<{column_length}}  {cd.column_type}"
        if column_name == schema.id_column:
            column_str += f", ID column (in ID space {schema.id_space})"
        if column_name == schema.grouping_column:
            column_str += ", grouping column"
        if not cd.allow_null:
            column_str += ", not null"
        if cd.column_type == ColumnType.DECIMAL:
            if not cd.allow_nan:
                column_str += ", not NaN"
            if not cd.allow_inf:
                column_str += ", not infinity"
        description.append(column_str)
    return description
