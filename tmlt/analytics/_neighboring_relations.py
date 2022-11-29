"""Module containing supported variants of neighboring relations."""
# pylint: disable=useless-super-delegation, protected-access, no-self-use

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from pyspark.sql import DataFrame
from pyspark.sql.types import DateType, IntegerType, LongType, StringType
from typeguard import check_type

from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail


class NeighboringRelation(ABC):
    """Base class for a NeighboringRelation."""

    @abstractmethod
    def validate_input(self, dfs: Dict[str, DataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception.

        Used only for top-level validation.

        Exception types and common reasons:
           - TypeError: Input dictionary does not map table names to Spark DataFrames
           - ValueError: Input dictionary contains an invalid number of items or
             contains invalid values.
           - KeyError: Relation table doesn't exist in input dictionary
        """

    @abstractmethod
    def _validate(self, dfs: Dict[str, DataFrame]) -> Any:
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.

        Returns a list of table names from the input.
        """

    @abstractmethod
    def accept(self, visitor) -> Any:
        """Returns the result of a visit to core for this relation."""


@dataclass(frozen=True)
class AddRemoveRows(NeighboringRelation):
    """A relation of tables differing by a limited number of rows.

    Two tables are considered neighbors under this relation if
    they differ by at most n rows.
    """

    table: str
    """The name of the table in this relation."""
    n: int = field(default=1)
    """The max number of rows wich may be differ for two instances of the table to
     be neighbors.
     """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type("source_id", self.table, str)
        check_type("n", self.n, int)

    def validate_input(self, dfs: Dict[str, DataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception.

        Used only for top-level validation.
        """
        check_type("dfs", dfs, Dict[str, DataFrame])
        if len(dfs) > 1:
            raise ValueError(
                f"The provided input contains too many items: {dfs.items()}."
                " The AddRemoveRows relation requires input with one item."
            )
        self._validate(dfs)
        return True

    def _validate(self, dfs: Dict[str, DataFrame]) -> List[str]:
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.
        """
        # validation checks that can be called by other relations. this
        # just verifies that the initialized table is in the dfs input,
        # and that it points to a dataframe object in the Dict.
        coerce_spark_schema_or_fail(dfs[self.table])
        if self.table not in dfs.keys():
            raise ValueError(
                f"""The provided input doesn't contain the relation table
                Input table names: {dfs.keys()}
                Relation table name: {self.table}"""
            )
        return [self.table]

    def accept(self, visitor: "NeighboringRelationVisitor") -> Any:
        """Visit this NeighboringRelation with a Visitor."""
        return visitor.visit_add_remove_rows(self)


@dataclass(frozen=True)
class AddRemoveRowsAcrossGroups(NeighboringRelation):
    """A relation of tables differing by a limited number of groups.

    Two tables are considered neighbors under this relation if they differ by
     at most n groups, with each group differing by no more than m rows.
    """

    table: str
    """The name of the table in this relation."""
    grouping_column: str
    """The column that must be grouped over for the privacy guarantee to hold."""
    max_groups: int
    """The maximum number of groups which may differ for two instances of the table
     to be neighbors.
    """
    per_group: Union[int, float]
    """The max number of rows in any single group that may differ for two instances of
     the table to be neighbors.
     """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type("table", self.table, str)
        check_type("grouping_column", self.grouping_column, str)
        check_type("max_groups", self.max_groups, int)
        check_type("per_group", self.per_group, (int, float))

    def validate_input(self, dfs: Dict[str, DataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception.

        Used only for top-level validation.
        """
        # checks to be done in any case: the table is in the relation,
        # and the columns exist and have appropriate types
        # private checks to be done if this is a top-level call:
        # the input dict is of length one, is a DataFrame + public checks.
        check_type("dfs", dfs, Dict[str, DataFrame])
        if len(dfs) > 1:
            raise ValueError(
                f"The provided input contains too many items: {dfs.items()}."
                " The AddRemoveRowsAcrossGroups relation requires input with one item."
            )

        self._validate(dfs)
        return True

    def _validate(self, dfs: Dict[str, DataFrame]) -> List[str]:
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.
        """
        # checks needed here is that the table is a DataFrame,
        # the grouping column exists
        # and is appropriately typed (i.e. is a supported type for grouping)
        if self.table not in dfs.keys():
            raise KeyError(
                f"""The provided input doesn't contain the relation table
                Input table names: {dfs.keys()}
                Relation table name: {self.table}"""
            )
        if self.grouping_column not in dfs[self.table].columns:
            raise ValueError(
                f"Grouping column '{self.grouping_column}' does not exist in the input."
                f" Available columns: {', '.join(dfs[self.table].columns)}"
            )

        allowed_types = [LongType(), StringType(), DateType()]
        coerced_df = coerce_spark_schema_or_fail(dfs[self.table])
        if any(
            (
                df_field.name == self.grouping_column
                and df_field.dataType not in allowed_types
            )
            for df_field in coerced_df.schema
        ):
            raise ValueError(
                f"Grouping column '{self.grouping_column}' is not of a type on which"
                " grouping is supported. Supported types for grouping:"
                f" {LongType(), StringType(), DateType(), IntegerType()}"
            )
        return [self.table]

    def accept(self, visitor: "NeighboringRelationVisitor") -> Any:
        """Visit this NeighboringRelation with a Visitor."""
        return visitor.visit_add_remove_rows_across_groups(self)


@dataclass(init=False, frozen=True)
class Conjunction(NeighboringRelation):
    """A conjunction composed of other neighboring relations."""

    children: List[NeighboringRelation]
    """Other neighboring relations to build the Conjunction.
     Args can be provided as a single list or as separate arguments.

     If more than one list is provided, Conjunction will only use the first.
     """

    def __init__(self, *children) -> None:
        """Constructor."""
        # flatten the (potentially) nested list
        # since frozen is set to True, we need to subvert it to flatten.
        if isinstance(children[0], list):
            object.__setattr__(self, "children", self._flatten(children[0]))
        else:
            object.__setattr__(self, "children", self._flatten(list(children)))

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("children", self.children, List[NeighboringRelation])

    def validate_input(self, dfs: Dict[str, DataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception."""
        # checks that the provided input maps tables names to DataFrames
        # checks that every input table in dfs is covered in the relation
        # checks each table is covered only once in the relation
        # validation checks pass for each of the children
        check_type("dfs", dfs, Dict[str, DataFrame])
        covered_tables: List[str] = []
        for child in self.children:
            relation_table_names = child._validate(dfs)
            # the sets have at least one common member, which means
            # the table(s) is/are already being used in the relation
            if set(covered_tables) & set(relation_table_names):
                raise ValueError(
                    f"""It appears a table is used more than once in the relation.
                    Duplicate table(s): {set(covered_tables)&
                    set(relation_table_names)}"""
                )
            covered_tables.extend(relation_table_names)
        # the sets don't match, meaning a table was left out of either the relation
        # or the input dict
        if set(dfs.keys()) ^ set(covered_tables):
            raise ValueError(
                f"""It appears that the list of input tables doesn't match the list of
                tables used in the relation.
                Tables that appear in only one list: {set(dfs.keys())^
                set(covered_tables)}
                """
            )
        return True

    def accept(self, visitor: "NeighboringRelationVisitor") -> Any:
        """Visit this NeighboringRelation with a Visitor."""
        return visitor.visit_conjunction(self)

    def _validate(self, dfs: Dict[str, DataFrame]):
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.
        """
        # not necessary for conjunction, since it's always a top-level call
        return

    def _flatten(self, children_list):
        """Recursively flatten a Conjunction's list of child NeighboringRelations."""
        flat_list = []
        for element in children_list:
            if isinstance(element, Conjunction):
                flat_list.extend(element._flatten(element.children))
            else:
                flat_list.append(element)
        return flat_list


class NeighboringRelationVisitor(ABC):
    """A base class for implementing visitors for :class:`NeighboringRelation`."""

    @abstractmethod
    def visit_add_remove_rows(self, relation: AddRemoveRows) -> Any:
        """Visit a :class:`AddRemoveRows`."""

    @abstractmethod
    def visit_add_remove_rows_across_groups(
        self, relation: AddRemoveRowsAcrossGroups
    ) -> Any:
        """Visit a :class:`AddRemoveRowsAcrossGroups`."""

    @abstractmethod
    def visit_conjunction(self, relation: Conjunction) -> Any:
        """Visit a :class:`Conjunction`."""
