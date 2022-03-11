"""Unit tests for catalog."""

# <placeholder: boilerplate>

import unittest
from typing import Optional

from parameterized import parameterized

from tmlt.analytics._catalog import Catalog, PrivateTable
from tmlt.analytics._schema import ColumnType, Schema


class TestCatalog(unittest.TestCase):
    """Unit tests for Catalog."""

    @parameterized.expand([("A",), (None,)])
    def test_add_private_source(self, grouping_column: Optional["str"]):
        """Add private source."""
        catalog = Catalog()
        catalog.add_private_source(
            source_id="private",
            col_types={"A": ColumnType.VARCHAR},
            stability=3,
            grouping_column=grouping_column,
        )

        self.assertEqual(len(catalog.tables), 1)
        private_table = catalog.tables["private"]
        self.assertIsInstance(private_table, PrivateTable)
        assert isinstance(private_table, PrivateTable)  # for mypy
        self.assertIs(private_table, catalog.private_table)
        self.assertEqual(private_table.source_id, "private")
        actual_schema = private_table.schema
        expected_schema = Schema({"A": "VARCHAR"}, grouping_column=grouping_column)
        self.assertEqual(actual_schema, expected_schema)
        self.assertEqual(private_table.stability, 3)

    def test_add_public_source(self):
        """Add public source."""
        catalog = Catalog()
        catalog.add_private_source(
            source_id="public", col_types={"A": "VARCHAR"}, stability=1
        )

        self.assertEqual(len(catalog.tables), 1)
        self.assertEqual(list(catalog.tables)[0], "public")
        self.assertEqual(catalog.tables["public"].source_id, "public")
        actual_schema = catalog.tables["public"].schema
        expected_schema = Schema({"A": "VARCHAR"})
        self.assertEqual(actual_schema, expected_schema)

    def test_add_private_view(self):
        """Add private view."""
        catalog = Catalog()
        catalog.add_private_view(
            source_id="private_view", col_types={"A": "VARCHAR"}, stability=1
        )

        self.assertEqual(len(catalog.tables), 1)
        self.assertEqual(list(catalog.tables)[0], "private_view")
        self.assertEqual(catalog.tables["private_view"].source_id, "private_view")
        actual_schema = catalog.tables["private_view"].schema
        expected_schema = Schema({"A": "VARCHAR"})
        self.assertEqual(actual_schema, expected_schema)
        self.assertEqual(catalog.tables["private_view"].stability, 1)

    def test_private_source_already_exists(self):
        """Add invalid private source"""
        catalog = Catalog()
        source_id = "private"
        catalog.add_private_source(
            source_id=source_id, col_types=({"A": "VARCHAR"}), stability=1
        )
        with self.assertRaisesRegex(
            RuntimeError, "Cannot have more than one private source"
        ):
            catalog.add_private_source(
                source_id=source_id, col_types={"B": "VARCHAR"}, stability=1
            )

    def test_invalid_addition_private_view(self):
        """Add invalid private view"""
        catalog = Catalog()
        source_id = "private"
        catalog.add_private_source(
            source_id=source_id, col_types={"A": "VARCHAR"}, stability=1
        )
        with self.assertRaisesRegex(
            ValueError, f"{source_id} already exists in catalog."
        ):
            catalog.add_private_view(
                source_id=source_id, col_types={"B": "VARCHAR"}, stability=1
            )

    def test_invalid_addition_public_source(self):
        """Add invalid public source"""
        catalog = Catalog()
        source_id = "public"
        catalog.add_public_source(source_id, {"A": "VARCHAR"})
        with self.assertRaisesRegex(
            ValueError, f"{source_id} already exists in catalog."
        ):
            catalog.add_public_source(source_id, {"C": "VARCHAR"})
