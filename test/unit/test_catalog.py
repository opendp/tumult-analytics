"""Unit tests for catalog."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Optional

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from tmlt.analytics._catalog import Catalog, PrivateTable
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema


@pytest.mark.parametrize("grouping_column", ["A", (None)])
def test_add_private_table(grouping_column: Optional["str"]):
    """Adding a private table works as expected."""
    catalog = Catalog()
    catalog.add_private_table(
        source_id="private",
        col_types={"A": ColumnDescriptor(ColumnType.VARCHAR)},
        constraints=[],
        grouping_column=grouping_column,
    )
    assert len(catalog.private_tables) == 1
    private_table = catalog.private_tables["private"]
    assert isinstance(private_table, PrivateTable)
    assert private_table.source_id == "private"
    assert private_table.constraints == tuple()
    actual_schema = private_table.schema
    expected_schema = Schema({"A": ColumnType.VARCHAR}, grouping_column=grouping_column)
    assert actual_schema == expected_schema


def test_add_public_table(spark):
    """Adding a public table works as expected."""
    catalog = Catalog()
    dataframe = spark.createDataFrame(
        [], schema=StructType([StructField("A", StringType(), True)])
    )
    catalog.add_public_table(
        source_id="public", col_types={"A": ColumnType.VARCHAR}, dataframe=dataframe
    )
    assert len(catalog.public_tables) == 1
    assert list(catalog.public_tables)[0] == "public"
    assert catalog.public_tables["public"].source_id == "public"
    actual_schema = catalog.public_tables["public"].schema
    expected_schema = Schema({"A": ColumnType.VARCHAR})
    assert actual_schema == expected_schema
    assert catalog.public_tables["public"].dataframe is dataframe


def test_invalid_addition_private_table():
    """Adding a private table that already exists fails."""
    catalog = Catalog()
    source_id = "private"
    catalog.add_private_table(
        source_id=source_id, col_types={"A": ColumnType.VARCHAR}, constraints=[]
    )
    with pytest.raises(
        ValueError, match=f"Table '{source_id}' already exists in catalog"
    ):
        catalog.add_private_table(
            source_id=source_id, col_types={"B": ColumnType.VARCHAR}, constraints=[]
        )


def test_invalid_addition_public_table(spark):
    """Adding a public table that already exists fails."""
    catalog = Catalog()
    source_id = "public"
    dataframe = spark.createDataFrame(
        [], schema=StructType([StructField("A", StringType(), True)])
    )
    catalog.add_public_table(source_id, {"A": ColumnType.VARCHAR}, dataframe)
    with pytest.raises(
        ValueError, match=f"Table '{source_id}' already exists in catalog"
    ):
        catalog.add_public_table(source_id, {"C": ColumnType.VARCHAR}, dataframe)
