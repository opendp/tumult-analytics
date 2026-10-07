"""Tests for building and caching the Session catalog."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from unittest.mock import patch

import pandas as pd
import pytest

from tmlt.analytics import (
    AddMaxRows,
    AddMaxRowsInMaxGroups,
    AddOneRow,
    AddRowsWithID,
    ApproxDPBudget,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerID,
    PureDPBudget,
    QueryBuilder,
    RhoZCDPBudget,
    Session,
)
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._table_identifier import NamedTable

BUDGETS = [
    PureDPBudget(float("inf")),
    ApproxDPBudget(float("inf"), 1),
    RhoZCDPBudget(float("inf")),
]
BUDGET_IDS = ["puredp", "approxdp", "zcdp"]


def _reference_catalog(session: Session) -> Catalog:
    """Builds a catalog the way Session._catalog did before it was optimized.

    This looks up every table separately, starting from the root of the
    session's input domain each time.
    """
    catalog = Catalog()
    for table in session.private_sources:
        catalog.add_private_table(
            table,
            session.get_schema(table),
            constraints=session._table_constraints[NamedTable(table)],
            grouping_column=session.get_grouping_column(table),
            id_column=session.get_id_column(table),
            id_space=session.get_id_space(table),
        )
    for table in session.public_sources:
        catalog.add_public_table(
            table,
            session.get_schema(table),
            session.public_source_dataframes[table],
        )
    return catalog


def _assert_catalogs_equal(actual: Catalog, expected: Catalog) -> None:
    """Checks that two catalogs have the same tables, in the same order."""
    assert list(actual.private_tables.items()) == list(expected.private_tables.items())
    assert list(actual.public_tables) == list(expected.public_tables)
    for name, expected_table in expected.public_tables.items():
        actual_table = actual.public_tables[name]
        assert actual_table.source_id == expected_table.source_id
        assert actual_table.schema == expected_table.schema
        assert actual_table.dataframe is expected_table.dataframe


def _assert_catalog_up_to_date(session: Session) -> None:
    """Checks that the session's catalog matches a freshly built one."""
    _assert_catalogs_equal(session._catalog, _reference_catalog(session))


@pytest.fixture(name="dataframes", scope="module")
def fixture_dataframes(spark):
    """Small dataframes for building sessions."""
    return {
        "ids": spark.createDataFrame(
            pd.DataFrame(
                [[1, "A", 4], [1, "B", 5], [2, "A", 6], [3, "B", 7]],
                columns=["id", "group", "n"],
            )
        ),
        "rows": spark.createDataFrame(
            pd.DataFrame([["A", 1.0], ["B", 2.0], ["A", 3.0]], columns=["group", "x"])
        ),
        "public": spark.createDataFrame(
            pd.DataFrame([["A", 10], ["B", 20]], columns=["group", "p"])
        ),
        "public2": spark.createDataFrame(
            pd.DataFrame({"group": ["A", "B"], "q": ["x", "y"]})
        ),
    }


def _build_session(budget, dataframes) -> Session:
    """A Session with ID spaces, grouping columns, and public tables."""
    return (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_id_space("space_a")
        .with_id_space("space_b")
        .with_private_dataframe(
            "id_a1", dataframes["ids"], protected_change=AddRowsWithID("id", "space_a")
        )
        .with_private_dataframe(
            "id_a2", dataframes["ids"], protected_change=AddRowsWithID("id", "space_a")
        )
        .with_private_dataframe(
            "id_b1", dataframes["ids"], protected_change=AddRowsWithID("id", "space_b")
        )
        .with_private_dataframe(
            "rows_one", dataframes["rows"], protected_change=AddOneRow()
        )
        .with_private_dataframe(
            "rows_max", dataframes["rows"], protected_change=AddMaxRows(2)
        )
        .with_private_dataframe(
            "rows_grouped",
            dataframes["rows"],
            protected_change=AddMaxRowsInMaxGroups(
                "group", max_groups=2, max_rows_per_group=1
            ),
        )
        .with_public_dataframe("public", dataframes["public"])
        .build()
    )


@pytest.mark.parametrize("budget", BUDGETS, ids=BUDGET_IDS)
def test_catalog_matches_reference(budget, dataframes):
    """The catalog is identical to one built with per-table lookups."""
    session = _build_session(budget, dataframes)
    catalog = session._catalog
    _assert_catalogs_equal(catalog, _reference_catalog(session))

    # Sanity-check that the session exercises every kind of catalog entry.
    private_tables = catalog.private_tables
    assert private_tables["id_a1"].schema.id_column == "id"
    assert private_tables["id_a1"].schema.id_space == "space_a"
    assert private_tables["id_b1"].schema.id_space == "space_b"
    assert private_tables["rows_one"].schema.id_space is None
    assert private_tables["rows_grouped"].schema.grouping_column == "group"
    assert list(catalog.public_tables) == ["public"]

    # Views with constraints, both cached and uncached, including IDs views.
    session.create_view(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(2)), "id_view", cache=True
    )
    session.create_view(
        QueryBuilder("rows_one").join_public("public"), "rows_view", cache=False
    )
    session.create_view(QueryBuilder("id_b1"), "id_view2", cache=False)
    session.add_public_dataframe("public2", dataframes["public2"])
    catalog = session._catalog
    _assert_catalogs_equal(catalog, _reference_catalog(session))
    assert catalog.private_tables["id_view"].constraints == (MaxRowsPerID(2),)
    assert catalog.private_tables["id_view"].schema.id_space == "space_a"
    assert "rows_view" in catalog.private_tables
    assert catalog.private_tables["id_view2"].schema.id_space == "space_b"
    assert list(catalog.public_tables) == ["public", "public2"]


@pytest.mark.parametrize("budget", BUDGETS, ids=BUDGET_IDS)
def test_catalog_not_rebuilt_per_query(budget, dataframes):
    """Queries reuse the cached catalog instead of rebuilding it."""
    session = _build_session(budget, dataframes)
    with patch.object(
        Session, "_build_catalog", autospec=True, side_effect=Session._build_catalog
    ) as build_catalog:
        first = session._catalog
        assert build_catalog.call_count == 1
        assert session._catalog is first

        session.evaluate(QueryBuilder("rows_one").count(), budget)
        session.evaluate(
            QueryBuilder("rows_max")
            .groupby(KeySet.from_dict({"group": ["A"]}))
            .count(),
            budget,
        )
        session.evaluate(QueryBuilder("id_a1").enforce(MaxRowsPerID(1)).count(), budget)
        session.describe()
        session.describe("id_a1")
        assert build_catalog.call_count == 1
        assert session._catalog is first


@pytest.mark.parametrize("budget", BUDGETS, ids=BUDGET_IDS)
def test_catalog_updated_after_mutations(budget, dataframes):
    """The catalog reflects every change to the session's tables."""
    session = _build_session(budget, dataframes)
    _assert_catalog_up_to_date(session)

    # add_public_dataframe
    session.add_public_dataframe("public2", dataframes["public2"])
    assert "public2" in session._catalog.public_tables
    _assert_catalog_up_to_date(session)
    session.evaluate(QueryBuilder("rows_one").join_public("public2").count(), budget)

    # create_view (uncached), with constraints
    session.create_view(
        QueryBuilder("id_a1").enforce(MaxRowsPerID(2)), "view1", cache=False
    )
    assert session._catalog.private_tables["view1"].constraints == (MaxRowsPerID(2),)
    _assert_catalog_up_to_date(session)
    # The new view can be queried immediately.
    session.evaluate(QueryBuilder("view1").count(), budget)

    # create_view (cached)
    session.create_view(QueryBuilder("rows_one"), "view2", cache=True)
    assert "view2" in session._catalog.private_tables
    _assert_catalog_up_to_date(session)
    session.evaluate(QueryBuilder("view2").count(), budget)

    # delete_view
    session.delete_view("view1")
    assert "view1" not in session._catalog.private_tables
    _assert_catalog_up_to_date(session)
    with pytest.raises(ValueError, match="view1"):
        session.evaluate(QueryBuilder("view1").count(), budget)

    # Recreating a deleted view with different constraints.
    session.create_view(QueryBuilder("id_a1"), "view1", cache=False)
    recreated_view = session._catalog.private_tables["view1"]
    assert not recreated_view.constraints
    _assert_catalog_up_to_date(session)

    session.delete_view("view2")
    assert "view2" not in session._catalog.private_tables
    _assert_catalog_up_to_date(session)


@pytest.mark.parametrize("budget", BUDGETS, ids=BUDGET_IDS)
def test_catalog_public_sources_changed_elsewhere(budget, dataframes):
    """Changes to the public sources made outside the session are picked up."""
    session = _build_session(budget, dataframes)
    first = session._catalog
    session.public_source_dataframes["public2"] = dataframes["public2"]
    assert session._catalog is not first
    assert list(session._catalog.public_tables) == ["public", "public2"]
    _assert_catalog_up_to_date(session)

    # Replacing a dataframe under the same name is also picked up.
    session.public_source_dataframes["public2"] = dataframes["public"]
    assert session._catalog.public_tables["public2"].dataframe is dataframes["public"]
    _assert_catalog_up_to_date(session)

    del session.public_source_dataframes["public2"]
    assert list(session._catalog.public_tables) == ["public"]
    _assert_catalog_up_to_date(session)


@pytest.mark.parametrize("budget", BUDGETS, ids=BUDGET_IDS)
@pytest.mark.parametrize(
    "constraint,expected_id_space",
    [(None, None), (MaxGroupsPerID("group", 1), "space_a"), (MaxRowsPerID(2), None)],
    ids=["rows", "ids_max_groups", "ids_max_rows"],
)
def test_catalog_after_partition_and_create(
    budget, constraint, expected_id_space, dataframes
):
    """Catalogs of partitioned sessions (and their parent) stay up to date."""
    session = _build_session(budget, dataframes)
    if constraint is None:
        source_id = "rows_one"
    else:
        source_id = "id_view"
        session.create_view(
            QueryBuilder("id_a1").enforce(constraint), source_id, cache=False
        )
    parent_catalog = session._catalog
    new_sessions = session.partition_and_create(
        source_id,
        privacy_budget=budget,
        column="group",
        splits={"part_a": "A", "part_b": "B"},
    )
    part_a = new_sessions["part_a"]
    part_b = new_sessions["part_b"]
    for child, name in [(part_a, "part_a"), (part_b, "part_b")]:
        assert list(child._catalog.private_tables) == [name]
        assert child._catalog.private_tables[name].schema.id_space == (
            expected_id_space
        )
        _assert_catalog_up_to_date(child)

    # Child sessions share their public sources with the parent and each other,
    # so adding a public table to one of them is visible in all of them.
    part_a.add_public_dataframe("public2", dataframes["public2"])
    for sess in [part_a, part_b]:
        assert list(sess._catalog.public_tables) == ["public", "public2"]
        _assert_catalog_up_to_date(sess)

    # Child sessions can create views as usual.
    part_b.create_view(QueryBuilder("part_b"), "part_b_view", cache=False)
    assert list(part_b._catalog.private_tables) == ["part_b_view", "part_b"]
    _assert_catalog_up_to_date(part_b)
    view_query = QueryBuilder("part_b_view")
    if expected_id_space is not None:
        view_query = view_query.enforce(MaxRowsPerID(1))
    part_b.evaluate(view_query.count(), budget)

    assert session._catalog is not parent_catalog
    _assert_catalog_up_to_date(session)
