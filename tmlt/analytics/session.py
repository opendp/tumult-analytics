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

# Copyright Tumult Labs 2022
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast
from warnings import warn

import pandas as pd  # pylint: disable=unused-import
import sympy as sp
from pyspark.sql import SparkSession  # pylint: disable=unused-import
from pyspark.sql import DataFrame
from typeguard import check_type, typechecked

from tmlt.analytics._catalog import Catalog
from tmlt.analytics._coerce_spark_schema import (
    SUPPORTED_SPARK_TYPES,
    TYPE_COERCION_MAP,
    coerce_spark_schema_or_fail,
)
from tmlt.analytics._neighboring_relation_visitor import NeighboringRelationCoreVisitor
from tmlt.analytics._neighboring_relations import (
    AddRemoveRows,
    AddRemoveRowsAcrossGroups,
    Conjunction,
    NeighboringRelation,
)
from tmlt.analytics._noise_info import _noise_from_measurement
from tmlt.analytics._privacy_budget_rounding_helper import get_adjusted_budget
from tmlt.analytics._query_expr_compiler import QueryExprCompiler
from tmlt.analytics._schema import (
    Schema,
    spark_dataframe_domain_to_analytics_columns,
    spark_schema_to_analytics_columns,
)
from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.protected_change import (
    AddMaxRows,
    AddMaxRowsInMaxGroups,
    AddOneRow,
    ProtectedChange,
)
from tmlt.analytics.query_builder import ColumnType, QueryBuilder
from tmlt.analytics.query_expr import QueryExpr
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
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.dictionary import (
    AugmentDictTransformation,
    CreateDictFromValue,
    GetValue,
    Subset,
    create_transform_value,
)
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.transformations.spark_transformations.persist import Persist, Unpersist
from tmlt.core.utils.configuration import SparkConfigError, check_java11
from tmlt.core.utils.exact_number import ExactNumber

__all__ = ["Session", "SUPPORTED_SPARK_TYPES", "TYPE_COERCION_MAP"]


class _PrivateSourceTuple(NamedTuple):
    """Named tuple of private Dataframe, domain and stability."""

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
        else:
            raise ValueError(
                f"Unsupported ProtectedChange type: {type(protected_change)}"
            )
    return Conjunction(relations)


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

        def build(self) -> "Session":
            """Builds Session with specified configuration."""
            if self._privacy_budget is None:
                raise ValueError("Privacy budget must be specified.")
            if not self._private_sources:
                raise ValueError("At least one private source must be provided.")

            output_measure: Union[PureDP, RhoZCDP]
            sympy_budget: sp.Expr
            if isinstance(self._privacy_budget, PureDPBudget):
                output_measure = PureDP()
                sympy_budget = ExactNumber.from_float(
                    self._privacy_budget.epsilon, round_up=False
                ).expr
            elif isinstance(self._privacy_budget, RhoZCDPBudget):
                output_measure = RhoZCDP()
                sympy_budget = ExactNumber.from_float(
                    self._privacy_budget.rho, round_up=False
                ).expr
            else:
                raise ValueError(
                    f"Unsupported PrivacyBudget variant: {type(self._privacy_budget)}"
                )

            neighboring_relation = _generate_neighboring_relation(self._private_sources)
            tables = {
                source_id: source_tuple.dataframe
                for source_id, source_tuple in self._private_sources.items()
            }
            domain, metric, distance, dataframes = neighboring_relation.accept(
                NeighboringRelationCoreVisitor(tables, output_measure)
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
            return Session(
                accountant=accountant,
                public_sources=self._public_sources,
                compiler=compiler,
            )

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
                stability: Use ``protected_change`` instead, see
                    :ref:`changelog<changelog#protected-change>`.
                grouping_column: Use ``protected_change`` instead, see
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

            # TODO(#2302): All paths through the below need deprecation
            #     warnings, for either the use of stability/grouping_column or
            #     the assumption of AddOneRow() if no stability is specified.
            if stability is None:
                if grouping_column is None:
                    protected_change = AddOneRow()
                else:
                    protected_change = AddMaxRowsInMaxGroups(grouping_column, 1, 1)
                    grouping_column = None
            else:
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
        if not isinstance(self._accountant.output_measure, (PureDP, RhoZCDP)):
            raise ValueError("Accountant is not using PureDP or RhoZCDP privacy.")
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
            stability: Use ``protected_change`` instead, see
                :ref:`changelog<changelog#protected-change>`
            grouping_column: Use ``protected_change`` instead, see
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
        return session_builder.build()

    @property
    def private_sources(self) -> List[str]:
        """Returns the ids of the private sources."""
        return list(self._input_domain.key_to_domain)

    @property
    def public_sources(self) -> List[str]:
        """Returns the ids of the public sources."""
        return list(self._public_sources)

    @property
    def public_source_dataframes(self) -> Dict[str, DataFrame]:
        """Returns a dictionary of public source dataframes."""
        return self._public_sources

    @property
    def remaining_privacy_budget(self) -> Union[PureDPBudget, RhoZCDPBudget]:
        """Returns the remaining privacy_budget left in the session.

        The type of the budget (e.g., PureDP or rho-zCDP) will be the same as
        the type of the budget the Session was initialized with.
        """
        sympy_budget = self._accountant.privacy_budget
        budget_value = ExactNumber(sympy_budget).to_float(round_up=False)
        output_measure = self._accountant.output_measure
        if output_measure == PureDP():
            return PureDPBudget(budget_value)
        if output_measure == RhoZCDP():
            return RhoZCDPBudget(budget_value)
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
        return cast(DictDomain, self._accountant.input_domain)

    @property
    def _input_metric(self) -> DictMetric:
        """Returns the input metric of the underlying accountant."""
        if not isinstance(self._accountant.input_metric, DictMetric):
            raise AssertionError(
                "Session accountant's input metric has an incorrect type. This is "
                "probably a bug; please let us know about it so we can "
                "fix it!"
            )
        return cast(DictMetric, self._accountant.input_metric)

    @property
    def _stability(self) -> Dict[str, sp.Expr]:
        """Returns the unmodified stability of the underlying PrivacyAccountant."""
        return {
            source_id: ExactNumber(d_in_i).expr
            for source_id, d_in_i in self._accountant.d_in.items()
        }

    @typechecked
    def get_schema(self, source_id: str) -> Schema:
        """Returns the schema for any data source.

        This includes information on whether the columns are nullable.

        Args:
            source_id: The ID for the data source whose column types
                are being retrieved.
        """
        # TODO(#2172):replace old source indexing method with use of new lookup feature
        if source_id in self._input_domain.key_to_domain:
            return Schema(
                spark_dataframe_domain_to_analytics_columns(
                    self._input_domain[source_id]
                )
            )
        else:
            return Schema(
                spark_schema_to_analytics_columns(
                    self.public_source_dataframes[source_id].schema
                )
            )

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
        try:
            if isinstance(self._input_metric[source_id], IfGroupedBy):
                inner_metric = cast(IfGroupedBy, self._input_metric[source_id])
                return inner_metric.column
            return None
        except KeyError as e:
            if source_id in self.public_sources:
                raise ValueError(
                    f"'{source_id}' does not have a grouping column, "
                    "because it is not a private table."
                )
            raise e

    @property
    def _catalog(self) -> Catalog:
        """Returns the catalog."""
        catalog = Catalog()
        primary_source_id = list(self.private_sources)[0]
        view_source_ids = [
            source_id
            for source_id in self.private_sources
            if source_id != primary_source_id
        ]
        catalog.add_private_source(
            source_id=primary_source_id,
            col_types=self.get_schema(primary_source_id),
            # Catalogs require an integral stability. The catalog is only used for query
            # validation, so using the incorrect stability (if the true stability is
            # non-integral) is ok.
            stability=int(self._stability[primary_source_id]),
            grouping_column=self.get_grouping_column(primary_source_id),
        )
        for view_source_id in view_source_ids:
            catalog.add_private_view(
                view_source_id,
                self.get_schema(view_source_id),
                # Catalogs require integral stability, see note above.
                int(self._stability[view_source_id]),
                self.get_grouping_column(view_source_id),
            )

        for public_source_id in self.public_sources:
            catalog.add_public_source(
                public_source_id,
                spark_schema_to_analytics_columns(
                    self.public_source_dataframes[public_source_id].schema
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
        if source_id in self._public_sources:
            raise ValueError(
                "This session already has a public source with the source_id"
                f" {source_id}"
            )
        dataframe = coerce_spark_schema_or_fail(dataframe)
        self._public_sources[source_id] = dataframe

    def _compile_and_get_budget(
        self, query_expr: QueryExpr, privacy_budget: PrivacyBudget
    ) -> Tuple[Measurement, ExactNumber]:
        """Pre-processing needed for evaluate() and _noise_info()."""
        check_type("query_expr", query_expr, QueryExpr)
        check_type("privacy_budget", privacy_budget, PrivacyBudget)

        self._validate_budget_type_matches_session(privacy_budget)
        if privacy_budget == PureDPBudget(0) or privacy_budget == RhoZCDPBudget(0):
            raise ValueError("You need a non-zero privacy budget to evaluate a query.")

        adjusted_budget = self._process_requested_budget(privacy_budget)

        measurement = self._compiler(
            queries=[query_expr],
            privacy_budget=adjusted_budget.expr,
            stability=self._stability,
            input_domain=self._input_domain,
            input_metric=self._input_metric,
            public_sources=self._public_sources,
            catalog=self._catalog,
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

        try:
            if not measurement.privacy_relation(self._accountant.d_in, adjusted_budget):
                raise ValueError(
                    "With these inputs and this privacy budget, "
                    "similar inputs will *not* produce similar outputs. "
                )
            try:
                answers = self._accountant.measure(measurement, d_out=adjusted_budget)
            except InsufficientBudgetError:
                approx_budget_needed = adjusted_budget.to_float(round_up=True)
                if not isinstance(self._accountant.privacy_budget, ExactNumber):
                    raise AssertionError(
                        "Unable to convert privacy budget of"
                        f" {self._accountant.privacy_budget} to float. This is probably"
                        " a bug; please let us know about it so we can fix it!"
                    )
                approx_budget_left = self._accountant.privacy_budget.to_float(
                    round_up=False
                )
                approx_diff = abs(
                    (self._accountant.privacy_budget - adjusted_budget).to_float(
                        round_up=True
                    )
                )
                raise RuntimeError(
                    "Cannot answer query without exceeding privacy budget: it needs"
                    f" approximately {approx_budget_needed:.3f}, but the remaining"
                    f" budget is approximately {approx_budget_left:.3f} (difference:"
                    f" {approx_diff:.3e})"
                )
            if len(answers) != 1:
                raise AssertionError(
                    "Expected exactly one answer, but got "
                    f"{len(answers)} answers instead. This is "
                    "probably a bug; please let us know about it so "
                    "we can fix it!"
                )
            return answers[0]
        except InactiveAccountantError:
            raise RuntimeError(
                "This session is no longer active. Either it was manually stopped "
                "with session.stop(), or it was stopped indirectly by the "
                "activity of other sessions. See partition_and_create "
                "for more information."
            )

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
            ['my_private_data', 'private_public_join']
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
        if source_id in self._input_domain.key_to_domain:
            raise ValueError(f"ID {source_id} already exists.")

        if isinstance(query_expr, QueryBuilder):
            query_expr = query_expr.query_expr

        if not isinstance(query_expr, QueryExpr):
            raise ValueError("query_expr must be of type QueryBuilder or QueryExpr.")
        transformation = self._compiler.build_transformation(
            query=query_expr,
            input_domain=self._input_domain,
            input_metric=self._input_metric,
            public_sources=self._public_sources,
            catalog=self._catalog,
        )
        if cache:
            transformation = transformation | Persist(
                domain=cast(SparkDataFrameDomain, transformation.output_domain),
                metric=transformation.output_metric,
            )

        dict_transformation = AugmentDictTransformation(
            transformation
            | CreateDictFromValue(
                input_domain=transformation.output_domain,
                input_metric=transformation.output_metric,
                key=source_id,
            )
        )

        # This is a transform-in-place against the privacy accountant
        self._accountant.transform_in_place(dict_transformation)

    def delete_view(self, source_id: str):
        """Deletes a view and decaches it if it was cached.

        Args:
            source_id: The name of the view.
        """
        if source_id not in self._input_domain.key_to_domain:
            raise ValueError(f"ID {source_id} does not exist.")
        self._activate_accountant()

        # Unpersist does nothing if the DataFrame isn't persisted
        domain = cast(SparkDataFrameDomain, self._input_domain.key_to_domain[source_id])
        metric = self._input_metric.key_to_metric[source_id]
        unpersist_source = create_transform_value(
            input_domain=cast(DictDomain, self._accountant.input_domain),
            input_metric=cast(DictMetric, self._accountant.input_metric),
            key=source_id,
            transformation=Unpersist(domain, metric),
            hint=lambda d_in, _: d_in,
        )
        self._accountant.transform_in_place(
            unpersist_source, d_out=self._accountant.d_in
        )

        transformation = Subset(
            input_domain=self._input_domain,
            input_metric=self._input_metric,
            keys=list(set(self._input_domain.key_to_domain.keys()) - {source_id}),
        )
        d_out = {k: v for k, v in self._accountant.d_in.items() if k != source_id}
        self._accountant.transform_in_place(transformation, d_out)

    # pylint: disable=line-too-long
    # TODO(#2199): remove the attr_name argument
    @typechecked
    def partition_and_create(
        self,
        source_id: str,
        privacy_budget: PrivacyBudget,
        column: Optional[str] = None,
        splits: Optional[Union[Dict[str, str], Dict[str, int]]] = None,
        attr_name: Optional[str] = None,
    ) -> Dict[str, "Session"]:
        """Returns new sessions from a partition mapped to split name/``source_id``.

        The type of privacy budget that you use must match the type your Session was
        initialized with (i.e., you cannot use a RhoZCDPBudget to partition your
        Session if the Session was created using a PureDPBudget, and vice versa).

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
            attr_name: Deprecated synonym for ``column``. Using the ``column`` argument
                is preferred.
        """
        # pylint: enable=line-too-long
        if column is None and attr_name is None:
            raise ValueError("Please specify a column using the column parameter")
        if column is not None and attr_name is not None:
            raise ValueError("You cannot specify both a column and an attr_name")
        if attr_name is not None:
            warn(
                "The attr_name argument is deprecated and will be removed in a future"
                " release",
                DeprecationWarning,
            )
            column = attr_name
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
        self._validate_budget_type_matches_session(privacy_budget)
        self._activate_accountant()

        transformation = GetValue(
            input_domain=self._input_domain,
            input_metric=self._input_metric,
            key=source_id,
        )
        d_mid = self._accountant.d_in[source_id]
        if not transformation.stability_relation(self._accountant.d_in, d_mid):
            raise ValueError(
                "This partition is unstable: close inputs will not produce "
                "close outputs."
            )
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
        except KeyError:
            raise KeyError(
                f"'{column}' not present in transformed dataframe's columns; "
                "schema of transformed dataframe is "
                f"{spark_dataframe_domain_to_analytics_columns(transformation_domain)}"
            )

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
        if (
            isinstance(transformation.output_metric, IfGroupedBy)
            and column != transformation.output_metric.column
        ):
            element_metric = transformation.output_metric
        else:
            element_metric = SymmetricDifference()
        partition_transformation = PartitionByKeys(
            input_domain=transformation_domain,
            input_metric=transformation.output_metric,
            use_l2=isinstance(self._compiler.output_measure, RhoZCDP),
            keys=[column],
            list_values=split_vals,
        )
        chained_partition = transformation | partition_transformation
        if transformation.stability_function(self._accountant.d_in) != d_mid:
            raise AssertionError(
                "Transformation's stability function does not match "
                "transformed data. This is probably a bug; please let us "
                "know about it so we can fix it!"
            )

        adjusted_budget = self._process_requested_budget(privacy_budget)

        try:
            new_accountants = self._accountant.split(
                chained_partition, privacy_budget=adjusted_budget
            )
        except InactiveAccountantError:
            raise RuntimeError(
                "This session is no longer active. Either it was manually stopped"
                "with session.stop(), or it was stopped indirectly by the "
                "activity of other sessions. See partition_and_create "
                "for more information."
            )
        except InsufficientBudgetError:
            approx_budget_needed = adjusted_budget.to_float(round_up=True)
            if not isinstance(self._accountant.privacy_budget, ExactNumber):
                raise AssertionError(
                    "Unable to convert privacy budget of"
                    f" {self._accountant.privacy_budget} to float. This is probably a"
                    " bug; please let us know about it so we can fix it!"
                )

            approx_budget_left = self._accountant.privacy_budget.to_float(
                round_up=False
            )
            approx_diff = abs(
                (self._accountant.privacy_budget - adjusted_budget).to_float(
                    round_up=True
                )
            )
            raise RuntimeError(
                "Cannot perform this partition without exceeding privacy budget: it"
                f" needs approximately {approx_budget_needed:.3f}, but the remaining"
                f" budget is approximately {approx_budget_left:.3f} (difference:"
                f" {approx_diff:.3e})"
            )

        for i, source in enumerate(new_sources):
            dict_transformation_wrapper = CreateDictFromValue(
                input_domain=transformation_domain,
                input_metric=element_metric,
                key=source,
            )
            new_accountants[i].queue_transformation(
                transformation=dict_transformation_wrapper
            )

        new_sessions = dict()
        for new_accountant, source in zip(new_accountants, new_sources):
            new_sessions[source] = Session(
                new_accountant, self._public_sources, self._compiler
            )
        return new_sessions

    def _process_requested_budget(self, privacy_budget: PrivacyBudget) -> ExactNumber:
        """Process the requested budget to accommodate floating point imprecision.

        Args:
            privacy_budget: The requested budget.
        """
        remaining_budget = self._accountant.privacy_budget
        if not isinstance(remaining_budget, ExactNumber):
            raise AssertionError(
                f"Cannot understand remaining budget of {remaining_budget}. This is"
                " probably a bug; please let us know about it so we can fix it!"
            )
        if isinstance(privacy_budget, PureDPBudget):
            return get_adjusted_budget(privacy_budget.epsilon, remaining_budget)
        elif isinstance(privacy_budget, RhoZCDPBudget):
            return get_adjusted_budget(privacy_budget.rho, remaining_budget)
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
        matches_zcdp = isinstance(output_measure, RhoZCDP) and isinstance(
            privacy_budget, RhoZCDPBudget
        )
        if not (matches_puredp or matches_zcdp):
            raise ValueError(
                "Your requested privacy budget type must match the type of the privacy"
                " budget your Session was created with."
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
