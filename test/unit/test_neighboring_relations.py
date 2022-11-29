"""Tests for Neighboring Relations"""
# pylint: disable=pointless-string-statement, no-self-use, no-member
from cmath import inf, pi

import pandas as pd
import pytest
import sympy as sp

from tmlt.analytics._neighboring_relation_visitor import NeighboringRelationCoreVisitor
from tmlt.analytics._neighboring_relations import (
    AddRemoveRows,
    AddRemoveRowsAcrossGroups,
    Conjunction,
)
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    DictMetric,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber


@pytest.fixture(name="test_data", scope="class")
def setup_test_data(request, spark) -> None:
    """Set up test data."""
    table1 = spark.createDataFrame(
        pd.DataFrame(
            [["0", 1, 0], ["1", 0, 1], ["1", 2, 1], ["1", 2, 0]],
            columns=["A", "B", "X"],
        )
    )
    request.cls.table1 = table1

    table2 = spark.createDataFrame(
        pd.DataFrame(
            [["0", 1, 0], ["1", 0, 1], ["1", 2, 1], ["2", 1, 0], ["0", 1, 2]],
            columns=["A", "B", "X"],
        )
    )
    request.cls.table2 = table2

    table3 = spark.createDataFrame(
        pd.DataFrame(
            [
                [float(inf), 1, 0],
                [float(-inf), 0, 1],
                [float(-inf), 2, 1],
                [float(pi), 1, 0],
                [float(-pi), 1, 2],
            ],
            columns=["A", "B", "X"],
        )
    )
    request.cls.table3 = table3
    request.cls.testdfsdict = {"table1": table1, "table2": table2}
    request.cls.testdfsdictgrouping = {
        "table1": table1,
        "table2": table2,
        "table3": table3,
    }


@pytest.mark.usefixtures("test_data")
class TestNeighboringRelations:
    """Tests for the AddRemoveRows NeighboringRelation."""

    def test_add_remove_rows_validation(self):
        """Test that validate works as expected for AddRemoveRows."""
        valid_dict = {"table1": self.table1}
        assert AddRemoveRows("table1", n=1).validate_input(valid_dict)
        # table doesn't exist in dict
        with pytest.raises(KeyError):
            AddRemoveRows("table2", n=1).validate_input(valid_dict)
        # dict is too 'long' for this relation
        with pytest.raises(ValueError):
            AddRemoveRows("table1", n=1).validate_input(self.testdfsdict)
        # table's value is of wrong type
        with pytest.raises(TypeError):
            AddRemoveRows("table1", n=1).validate_input(
                {"table1": ["a", "random", "list"]}
            )

    def test_add_remove_rows_accept(self):
        """Tests that accept works as expected for AddRemoveRows."""
        pure_visitor = NeighboringRelationCoreVisitor(
            self.testdfsdict, output_measure=PureDP()
        )
        assert AddRemoveRows("table1", n=5).accept(pure_visitor) == (
            SparkDataFrameDomain.from_spark_schema(self.table1.schema),
            SymmetricDifference(),
            ExactNumber(5),
            self.table1,
        )

        rho_visitor = NeighboringRelationCoreVisitor(
            self.testdfsdict, output_measure=RhoZCDP()
        )
        assert AddRemoveRows("table2", n=5).accept(rho_visitor) == (
            SparkDataFrameDomain.from_spark_schema(self.table2.schema),
            SymmetricDifference(),
            ExactNumber(5),
            self.table2,
        )

    def test_add_remove_rows_across_groups_validation(self):
        """Tests that validate_input works as expected for AddRemoveRowsAcrossGroups"""
        assert AddRemoveRowsAcrossGroups("table1", "A", 1, 1).validate_input(
            {"table1": self.table1}
        )
        # table doesn't exist in dict
        with pytest.raises(KeyError):
            AddRemoveRowsAcrossGroups("table2", "A", 1, 1).validate_input(
                {"table1": self.table1}
            )
        # input dict contains too many elements
        with pytest.raises(ValueError):
            AddRemoveRowsAcrossGroups("table1", "A", 1, 1).validate_input(
                self.testdfsdictgrouping
            )
        # table's value is not a DataFrame
        with pytest.raises(TypeError):
            AddRemoveRowsAcrossGroups("table1", "B", 1, 1).validate_input(
                {"table1": ["a", "random", "list"]}
            )
        # table contains values not supported in grouping operations
        with pytest.raises(ValueError):
            AddRemoveRowsAcrossGroups("table3", "A", 1, 1).validate_input(
                {"table3": self.table3}
            )
        # grouping column doesn't exist in table
        with pytest.raises(ValueError):
            AddRemoveRowsAcrossGroups("table1", "Q", 1, 1).validate_input(
                {"table1": self.table1}
            )

    def test_add_remove_rows_across_groups_accept(self):
        """Tests that accept works as expected for AddRemoveRowsAcrossGroups"""
        pure_visitor = NeighboringRelationCoreVisitor(
            self.testdfsdict, output_measure=PureDP()
        )
        rho_visitor = NeighboringRelationCoreVisitor(self.testdfsdict, RhoZCDP())

        assert AddRemoveRowsAcrossGroups("table2", "B", 2, 3).accept(pure_visitor) == (
            SparkDataFrameDomain.from_spark_schema(self.table2.schema),
            IfGroupedBy("B", SumOf(SymmetricDifference())),
            ExactNumber(2 * 3),
            self.table2,
        )
        assert AddRemoveRowsAcrossGroups("table1", "A", 2, 3).accept(rho_visitor) == (
            SparkDataFrameDomain.from_spark_schema(self.table1.schema),
            IfGroupedBy("A", RootSumOfSquared(SymmetricDifference())),
            ExactNumber(sp.sqrt(2) * 3),
            self.table1,
        )

    #### Tests for Conjunction ####

    def test_conjunction_initialization(self):
        """Tests that initialization of Conjunction works as expected."""
        # conjunction with nested conjunctions. Should be flattened to a single
        # parent conjunction
        conjunction = Conjunction(
            AddRemoveRowsAcrossGroups("table3", "A", 1, 1),
            Conjunction(
                AddRemoveRows("table1", n=1),
                Conjunction(AddRemoveRows("table2", n=1), AddRemoveRows("table2", n=1)),
            ),
        )
        assert conjunction.children == [
            AddRemoveRowsAcrossGroups("table3", "A", 1, 1),
            AddRemoveRows("table1", n=1),
            AddRemoveRows("table2", n=1),
            AddRemoveRows("table2", n=1),
        ]

    def test_conjunction_accept(self):
        """Tests that accept method of Conjunction works as expected."""
        visitor = NeighboringRelationCoreVisitor(self.testdfsdict, PureDP())
        expected_add_remove_rows_result = AddRemoveRows("table1", 1).accept(visitor)
        expected_add_remove_rows_groups_result = AddRemoveRowsAcrossGroups(
            "table2", "A", 2, 1
        ).accept(visitor)
        assert Conjunction(
            AddRemoveRows("table1", 1), AddRemoveRowsAcrossGroups("table2", "A", 2, 1)
        ).accept(visitor) == (
            DictDomain(
                {
                    "table1": expected_add_remove_rows_result[0],
                    "table2": expected_add_remove_rows_groups_result[0],
                }
            ),
            DictMetric(
                {
                    "table1": expected_add_remove_rows_result[1],
                    "table2": expected_add_remove_rows_groups_result[1],
                }
            ),
            {
                "table1": expected_add_remove_rows_result[2],
                "table2": expected_add_remove_rows_groups_result[2],
            },
            {
                "table1": expected_add_remove_rows_result[3],
                "table2": expected_add_remove_rows_groups_result[3],
            },
        )

    def test_conjunction_validation(self):
        """Tests that validate_input works as expected for Conjunction."""
        assert Conjunction(
            AddRemoveRowsAcrossGroups("table1", "A", 1, 1), AddRemoveRows("table2", n=1)
        ).validate_input({"table1": self.table1, "table2": self.table2})
        # duplicate table name usage should throw exception
        with pytest.raises(ValueError):
            Conjunction(
                AddRemoveRowsAcrossGroups("table1", "A", 1, 1),
                AddRemoveRows("table2", n=1),
                AddRemoveRows("table2", n=1),
            ).validate_input({"table1": self.table1, "table2": self.table2})
        # not every table is used in the relation
        with pytest.raises(ValueError):
            Conjunction(
                AddRemoveRowsAcrossGroups("table1", "A", 1, 1),
                AddRemoveRows("table2", n=1),
            ).validate_input(
                {"table1": self.table1, "table2": self.table2, "table3": self.table3}
            )
