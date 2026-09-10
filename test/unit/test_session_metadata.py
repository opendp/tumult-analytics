"""Unit test for the data-invariance of Session metadata methods."""

import difflib
from typing import Any, Callable, Dict, List, Tuple

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from tmlt.analytics import (
    AddOneRow,
    AddRowsWithID,
    KeySet,
    ProtectedChange,
    PureDPBudget,
    QueryBuilder,
    Session,
)

# Two tables with the same schema (including nullability) but different values.
_INVARIANCE_SCHEMA = StructType(
    [
        StructField("id", StringType(), True),
        StructField("group", LongType(), True),
        StructField("value", DoubleType(), True),
    ]
)
_INVARIANCE_ROWS_PLAIN = [("x", 1, 1.0), ("y", 2, 2.0), ("x", 3, 3.0)]
_INVARIANCE_ROWS_HOSTILE = [
    (None, None, None),
    ("zzz", 10**9, float("nan")),
    ("q", -5, float("inf")),
] * 40


def _read_describe(session: Session, capsys: pytest.CaptureFixture) -> str:
    """Capture both forms of ``Session.describe``."""
    capsys.readouterr()  # discard anything buffered so far
    session.describe()
    described = capsys.readouterr().out
    # Group on B so that this works whether or not A is an ID column.
    session.describe(
        QueryBuilder("private").groupby(KeySet.from_dict({"group": [1, 2]})).count()
    )
    return described + capsys.readouterr().out


# Every public Session method that reports something, along with how to run it.
_METADATA_METHODS: Dict[str, Callable[[Session, Any], str]] = {
    "get_schema": lambda session, _: repr(session.get_schema("private")),
    "get_column_types": lambda session, _: repr(session.get_column_types("private")),
    "get_grouping_column": lambda session, _: repr(
        session.get_grouping_column("private")
    ),
    "get_id_column": lambda session, _: repr(session.get_id_column("private")),
    "get_id_space": lambda session, _: repr(session.get_id_space("private")),
    "private_sources": lambda session, _: repr(session.private_sources),
    "public_sources": lambda session, _: repr(session.public_sources),
    # Compared by name only: the values are the caller's own DataFrames, whose
    # reprs embed object identity and so differ between any two Sessions.
    "public_source_dataframes": lambda session, _: repr(
        sorted(session.public_source_dataframes)
    ),
    "remaining_privacy_budget": lambda session, _: repr(
        session.remaining_privacy_budget
    ),
    "describe": _read_describe,
}

# Public Session methods that either vary depending on the data (because they spend
# privacy budget), or mutate the Session.
_NON_METADATA_METHODS: List[str] = [
    "evaluate",  # consumes budget
    "partition_and_create",  # consumes budget
    "stop",  # retires the session
    "add_public_dataframe",  # mutates the session
    "create_view",  # mutates the session
    "delete_view",  # mutates the session
    "from_dataframe",  # constructor
    "Builder",  # constructor
]


def _session_metadata(
    rows: List[Tuple], protected_change: ProtectedChange, capsys: Any
) -> Dict[str, str]:
    """Read every swept Session surface for a Session built over ``rows``."""
    spark = SparkSession.builder.getOrCreate()
    session = Session.from_dataframe(
        privacy_budget=PureDPBudget(1),
        source_id="private",
        dataframe=spark.createDataFrame(rows, _INVARIANCE_SCHEMA),
        protected_change=protected_change,
    )
    return {name: reader(session, capsys) for name, reader in _METADATA_METHODS.items()}


@pytest.mark.parametrize(
    "protected_change",
    [AddOneRow(), AddRowsWithID("id")],
    ids=["add_one_row", "add_rows_with_id"],
)
def test_metadata_data_invariant(protected_change: ProtectedChange, capsys):
    """Public metadata methods shouldn't change depending on the data.

    Two sessions are built over identical declared schemas but very different contents,
    then we check all the metadata methods to make sure that they return the same thing.
    We also check that no "unexpected" methods are exposed in the Session: if someone
    later adds a method to the Session, this test fails. Then they must classify this
    method as either a metadata method (and test its data-invariance), or another method
    (and include it in _NON_METADATA_METHODS).
    """
    overlap = set(_METADATA_METHODS) & set(_NON_METADATA_METHODS)
    assert not overlap, f"Methods both metadata and non-metadata: {sorted(overlap)}"

    public_methods = {name for name in dir(Session) if not name.startswith("_")}
    classified = set(_METADATA_METHODS) | set(_NON_METADATA_METHODS)

    unclassified = public_methods - classified
    assert not unclassified, (
        f"New public Session methods {sorted(unclassified)} are not classified "
        "for the metadata-invariance test. If the member reports something to "
        "the caller, add it to _METADATA_METHODS so it gets checked; if it does not, "
        "add it to _NON_METADATA_METHODS."
    )

    stale = classified - public_methods
    assert not stale, (
        f"Session member(s) {sorted(stale)} no longer exist; remove them from "
        "_METADATA_METHODS / _NON_METADATA_METHODS."
    )

    plain = _session_metadata(_INVARIANCE_ROWS_PLAIN, protected_change, capsys)
    hostile = _session_metadata(_INVARIANCE_ROWS_HOSTILE, protected_change, capsys)

    assert plain.keys() == hostile.keys()
    differing = [name for name in plain if plain[name] != hostile[name]]
    assert not differing, (
        "Session metadata methods depend on private data values:\n\n"
        + "\n\n".join(
            f"--- {name} ---\n"
            + "\n".join(
                difflib.unified_diff(
                    plain[name].splitlines(),
                    hostile[name].splitlines(),
                    fromfile="plain data",
                    tofile="hostile data",
                    lineterm="",
                )
            )
            for name in differing
        )
    )
