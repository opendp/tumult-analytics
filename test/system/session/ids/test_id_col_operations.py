"""Integration tests for operations directly on ID columns."""

from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
from tmlt.core.domains.collections import DictDomain
from tmlt.core.metrics import AddRemoveKeys

from tmlt.analytics._table_identifier import TableCollection
from tmlt.analytics.constraints import MaxRowsPerID
from tmlt.analytics.protected_change import AddRowsWithID
from tmlt.analytics.query_builder import ColumnType, QueryBuilder
from tmlt.analytics.session import Session

from ..conftest import INF_BUDGET, INF_BUDGET_ZCDP


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,condition,expected_count",
    [
        (QueryBuilder("id_a1"), "id > 1", 3),
        (QueryBuilder("id_a1"), "id >= 1", 6),
        (QueryBuilder("id_a1"), "id < 3", 4),
        (QueryBuilder("id_a1"), "id > 3", 0),
    ],
)
def test_filter_on_id_col(
    query: QueryBuilder, condition: str, expected_count: int, session
):
    """Tests that filter on an ID column works as expected."""
    res = session.evaluate(
        query.enforce(MaxRowsPerID(100)).filter(condition).count(),
        session.remaining_privacy_budget,
    ).toPandas()
    assert res["count"][0] == expected_count


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,mapping,expected_sum",
    [
        (QueryBuilder("id_a1"), lambda row: {"new": row["id"]}, 11),
        (QueryBuilder("id_a1"), lambda row: {"new": row["id"] * 2}, 22),
        (
            QueryBuilder("id_a1"),
            lambda row: {"new": row["id"] if row["id"] == 4 else 0},
            0,
        ),
        (
            QueryBuilder("id_a1"),
            lambda row: {"new": row["id"] if row["id"] == 1 else 0},
            3,
        ),
    ],
)
def test_map_on_id_col(query: QueryBuilder, mapping: Any, expected_sum: int, session):
    """Tests that map on an ID column works as expected."""
    res = session.evaluate(
        query.enforce(MaxRowsPerID(100))
        .map(mapping, {"new": ColumnType.INTEGER}, augment=True)
        .sum("new", low=0, high=100),
        session.remaining_privacy_budget,
    ).toPandas()
    assert res["new_sum"][0] == expected_sum


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,replace_with",
    [(QueryBuilder("id_a3"), {"id": 100}), (QueryBuilder("id_a2"), {"id": 100})],
)
def test_replace_null_and_nan_raises_error(
    query: QueryBuilder, replace_with: Union[Dict[str, int], None], session
):
    """Tests that replace nulls/nans on an ID column raises an error."""
    with pytest.raises(ValueError, match="Cannot replace null values in column"):
        session.evaluate(
            query.enforce(MaxRowsPerID(100)).replace_null_and_nan(replace_with).count(),
            session.remaining_privacy_budget,
        )


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query", [QueryBuilder("id_a1"), QueryBuilder("id_a2"), QueryBuilder("id_a3")]
)
def test_replace_null_and_nan_raises_warning(session, query: QueryBuilder):
    """Tests that replace nulls/nans raises warning on IDs table with empty mapping."""
    with pytest.raises(
        RuntimeWarning, match="the ID column may still contain null values."
    ):
        session.evaluate(
            query.enforce(MaxRowsPerID(100)).replace_null_and_nan(None).count(),
            session.remaining_privacy_budget,
        )


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query", [QueryBuilder("id_a1"), QueryBuilder("id_a3"), QueryBuilder("id_a4")]
)
def test_drop_null_and_nan_raises_warning(session, query: QueryBuilder):
    """Tests that replace nulls/nans raises warning on IDs table with empty list."""
    with pytest.raises(
        RuntimeWarning, match="the ID column may still contain null values."
    ):
        session.evaluate(
            query.enforce(MaxRowsPerID(100)).drop_null_and_nan(None).count(),
            session.remaining_privacy_budget,
        )


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,columns",
    [
        (QueryBuilder("id_a1"), ["id"]),
        (QueryBuilder("id_a2"), ["id", "x"]),
        (QueryBuilder("id_a3"), ["id"]),
        (QueryBuilder("id_a3"), ["id", "x"]),
    ],
)
def test_drop_null_and_nan_raises_error(
    session, query: QueryBuilder, columns: Union[List[str], None]
):
    """Tests that replace nulls/nans raises warning on IDs table with empty mapping."""
    with pytest.raises(ValueError, match="it is an ID column."):
        session.evaluate(
            query.enforce(MaxRowsPerID(100)).drop_null_and_nan(columns).count(),
            session.remaining_privacy_budget,
        )


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,replace_with",
    [
        (QueryBuilder("id_a1"), {"id": (0, 0)}),
        (QueryBuilder("id_a3"), {"id": (0, 0)}),
        (QueryBuilder("id_a3"), {"id": (0, 0), "x": (0, 0)}),
    ],
)
def test_replace_infs_raises_error(
    session, query: QueryBuilder, replace_with: Optional[Dict[str, Tuple[float, float]]]
):
    """Tests that appropriate error is raised with replace infs on ID columns."""
    with pytest.raises(ValueError, match="Cannot replace infinite values in column"):
        session.evaluate(
            query.enforce(MaxRowsPerID(100)).replace_infinity(replace_with).count(),
            session.remaining_privacy_budget,
        )


@pytest.mark.parametrize(
    "session", [INF_BUDGET, INF_BUDGET_ZCDP], indirect=True, ids=["puredp", "zcdp"]
)
@pytest.mark.parametrize(
    "query,columns",
    [
        (QueryBuilder("id_a1"), ["id"]),
        (QueryBuilder("id_a3"), ["id"]),
        (QueryBuilder("id_a3"), ["id", "x"]),
    ],
)
def test_drop_infs_raises_error(session, query: QueryBuilder, columns: List[str]):
    """Tests that appropriate error is raised with drop infs on ID columns."""
    with pytest.raises(ValueError, match="Cannot drop infinite values in column"):
        session.evaluate(
            query.enforce(MaxRowsPerID(100)).drop_infinity(columns).count(),
            session.remaining_privacy_budget,
        )


@pytest.mark.parametrize("budget_type", [INF_BUDGET, INF_BUDGET_ZCDP])
@pytest.mark.parametrize(
    "df1,df2,df3,df4,nullable_id_in_a,nullable_id_in_b",
    [
        ("id1", "id2", "id3", "id4", True, True),
        ("id1", "id2", "id4", "id4", True, False),
        ("id4", "id4", "id3", "id4", False, True),
        ("id4", "id4", "id4", "id4", False, False),
        ("id1", "id4", "id3", "id4", True, True),
    ],
)
def test_various_session_builds(
    _session_data,
    budget_type,
    df1: str,
    df2: str,
    df3: str,
    df4: str,
    nullable_id_in_a: bool,
    nullable_id_in_b: bool,
):
    """Tests that various session builds work as expected."""
    sess = (
        Session.Builder()
        .with_privacy_budget(budget_type)
        .with_id_space("a")
        .with_id_space("b")
        .with_private_dataframe(
            "id_a1", _session_data[df1], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_a2", _session_data[df2], protected_change=AddRowsWithID("id", "a")
        )
        .with_private_dataframe(
            "id_b1", _session_data[df3], protected_change=AddRowsWithID("id", "b")
        )
        .with_private_dataframe(
            "id_b2", _session_data[df4], protected_change=AddRowsWithID("id", "b")
        )
    ).build()
    # pylint: disable=protected-access
    for table_collection, ark_metric in sess._input_metric.key_to_metric.items():
        dict_domain = sess._input_domain.key_to_domain[table_collection]
        assert isinstance(ark_metric, AddRemoveKeys)
        assert isinstance(dict_domain, DictDomain)
        sparkdf_domain_dict: Dict[Any, Any]
        sparkdf_domain_dict = dict_domain.key_to_domain
        nullable_id_in_domain = (
            nullable_id_in_a
            if table_collection == TableCollection("a")
            else nullable_id_in_b
        )
        if nullable_id_in_domain:
            assert all(
                sparkdf_domain_dict[table_id]  # type: ignore
                .schema[key_column]
                .allow_null
                for table_id, key_column in ark_metric.df_to_key_column.items()
            )
        else:
            assert not any(
                sparkdf_domain_dict[table_id]  # type: ignore
                .schema[key_column]
                .allow_null
                for table_id, key_column in ark_metric.df_to_key_column.items()
            )
    # pylint: enable=protected-access
