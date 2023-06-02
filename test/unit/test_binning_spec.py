"""Unit tests for BinningSpec."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import datetime
from typing import Any, List

import pytest

from tmlt.analytics._schema import ColumnDescriptor
from tmlt.analytics.binning_spec import BinningSpec, _default_bin_names, _edges_as_str
from tmlt.analytics.query_builder import ColumnType


@pytest.mark.parametrize(
    "bin_edges,expected_strs",
    [
        ([0, 1, 2], ["0", "1", "2"]),
        (["0", "1", "2"], ["'0'", "'1'", "'2'"]),
        (
            [datetime.date(2022, 1, 1), datetime.date(2022, 2, 1)],
            ["2022-01-01", "2022-02-01"],
        ),
        ([0.0, 0.1, 0.2], ["0.00", "0.10", "0.20"]),
        ([0.0, 0.111111, 0.222222], ["0.00", "0.11", "0.22"]),
        ([0.0, 0.000001, 0.000002], ["0.000000", "0.000001", "0.000002"]),
        ([0.0, 1.000001], ["0.00", "1.00"]),
        ([0.0, 0.999, 2.0], ["0.00", "1.00", "2.00"]),
        ([0.0, 0.999, 1.0], ["0.000", "0.999", "1.000"]),
        (
            [datetime.datetime(2022, 1, 1, 0), datetime.datetime(2022, 2, 1, 5)],
            ["2022-01-01 00:00", "2022-02-01 05:00"],
        ),
        (
            [
                datetime.datetime(2022, 1, 1, 0),
                datetime.datetime(2022, 2, 1, 5, 30, 15, 20000),
            ],
            ["2022-01-01 00:00:00.000", "2022-02-01 05:30:15.020"],
        ),
        (
            [
                datetime.datetime(2022, 1, 1, 0),
                datetime.datetime(2022, 2, 1, 5, 30, 15, 1),
            ],
            ["2022-01-01 00:00:00.000000", "2022-02-01 05:30:15.000001"],
        ),
    ],
)
def test_edges_as_str(bin_edges: List, expected_strs: List[str]):
    """Conversion of bin edges to strings works as expected."""
    assert _edges_as_str(bin_edges) == expected_strs


@pytest.mark.parametrize(
    "args,expected_strs",
    [
        (([0, 1, 2], True, True), ["[0, 1]", "(1, 2]"]),
        (([0, 1, 2], False, True), ["[0, 1)", "[1, 2]"]),
        (([0, 1, 2], True, False), ["(0, 1]", "(1, 2]"]),
        (([0, 1, 2], False, False), ["[0, 1)", "[1, 2)"]),
        (
            ([datetime.date(2022, 1, 1), datetime.date(2022, 3, 15)], True, False),
            ["(2022-01-01, 2022-03-15]"],
        ),
    ],
)
def test_default_bin_names(args: List[Any], expected_strs: List[str]):
    """Generation of bin names from bin edges works as expected."""
    assert _default_bin_names(*args) == expected_strs


@pytest.mark.parametrize(
    "edges,ty",
    [
        ([0, 1], ColumnType.INTEGER),
        ([0.0, 1.0], ColumnType.DECIMAL),
        (["0", "1"], ColumnType.VARCHAR),
        ([datetime.date(2022, 1, 1), datetime.date(2022, 1, 2)], ColumnType.DATE),
        (
            [datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2)],
            ColumnType.TIMESTAMP,
        ),
    ],
)
def test_input_type(edges: List[Any], ty: ColumnType):
    """BinningSpec.input_type works as expected."""
    spec = BinningSpec(edges)
    assert spec.input_type == ty


@pytest.mark.parametrize(
    "names,ty",
    [
        ([0, 1], ColumnType.INTEGER),
        ([0.0, 1.0], ColumnType.DECIMAL),
        (["0", "1"], ColumnType.VARCHAR),
        ([datetime.date(2022, 1, 1), datetime.date(2022, 1, 2)], ColumnType.DATE),
        (
            [datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2)],
            ColumnType.TIMESTAMP,
        ),
    ],
)
def test_column_type(names: List[Any], ty: ColumnType):
    """BinningSpec.column_descriptor.column_type works as expected."""
    spec = BinningSpec([0, 1, 2], names=names)
    assert spec.column_descriptor.column_type == ty


def test_descriptor_allow_nan() -> None:
    """BinningSpec sets allow_nan as expected."""
    edges = [0, 5, 10]
    spec = BinningSpec(edges, names=[float("NaN"), float("0")])
    assert spec.column_descriptor.allow_nan
    spec = BinningSpec(edges, names=[float("-NaN"), float("0")])
    assert spec.column_descriptor.allow_nan
    spec = BinningSpec(edges, names=[float("0"), float("5")])
    assert not spec.column_descriptor.allow_nan
    spec = BinningSpec(edges, names=[float("0"), float(5)], nan_bin=float("nan"))
    assert spec.column_descriptor.allow_nan
    spec = BinningSpec(edges, names=[float("0"), float(5)], nan_bin=3.3)
    assert not spec.column_descriptor.allow_nan


def test_descriptor_allow_null() -> None:
    """BinningSpec sets allow_null as expected."""
    edges = [0, 5, 10]
    spec = BinningSpec(edges, names=["null", "5"])
    assert spec.column_descriptor.allow_null
    spec = BinningSpec(edges, names=["NULL", "5"])
    assert spec.column_descriptor.allow_null
    spec = BinningSpec(edges, names=["Null", "5"])
    assert spec.column_descriptor.allow_null
    spec = BinningSpec(edges, names=["0", "5"])
    assert spec.column_descriptor.allow_null


def test_descriptor_allow_inf() -> None:
    """BinningSpec sets allow_inf as expected."""
    edges = [float("0"), float("5"), float("10")]
    spec = BinningSpec(edges, names=[float(0), float("inf")])
    assert spec.column_descriptor.allow_inf
    spec = BinningSpec(edges, names=[float(0), float("-inf")])
    assert spec.column_descriptor.allow_inf
    spec = BinningSpec(edges, names=[float(0), float(5)])
    assert not spec.column_descriptor.allow_inf
    spec = BinningSpec(edges, names=[float(0), float("nan")], nan_bin=float("inf"))
    assert spec.column_descriptor.allow_inf
    spec = BinningSpec(edges, names=[float(0), float("nan")], nan_bin=float("-inf"))
    assert spec.column_descriptor.allow_inf
    spec = BinningSpec(edges, names=[float(0), float(5)], nan_bin=3.3)
    assert not spec.column_descriptor.allow_inf


@pytest.mark.parametrize(
    "spec,bins",
    [
        (BinningSpec([0, 1, 2]), ["[0, 1]", "(1, 2]"]),
        (BinningSpec([0.0, 1.0, 2.0]), ["[0.00, 1.00]", "(1.00, 2.00]"]),
        (BinningSpec(["0", "1", "2"]), ["['0', '1']", "('1', '2']"]),
        (BinningSpec([0, 1, 2], nan_bin="NaN"), ["[0, 1]", "(1, 2]", "NaN"]),
        (
            BinningSpec([0.0, 1.0, 2.0], nan_bin="NaN"),
            ["[0.00, 1.00]", "(1.00, 2.00]", "NaN"],
        ),
        (
            BinningSpec(["0", "1", "2"], nan_bin="NaN"),
            ["['0', '1']", "('1', '2']", "NaN"],
        ),
        (
            BinningSpec(["0", "1", "2"], nan_bin="['0', '1']"),
            ["['0', '1']", "('1', '2']"],
        ),
    ],
)
def test_bins(spec: BinningSpec, bins: List) -> None:
    """BinningSpec.bins() returns the expected value."""
    assert spec.bins() == bins
    assert spec.bins(include_null=True) == bins + [None]


def test_binning() -> None:
    """Basic BinningSpec works as expected."""
    spec = BinningSpec([0, 5, 10, 15, 20])
    assert spec.bins() == ["[0, 5]", "(5, 10]", "(10, 15]", "(15, 20]"]
    assert spec.bins(include_null=True) == [
        "[0, 5]",
        "(5, 10]",
        "(10, 15]",
        "(15, 20]",
        None,
    ]
    assert spec.column_descriptor == ColumnDescriptor(
        ColumnType.VARCHAR, allow_null=True, allow_nan=False, allow_inf=False
    )
    bin_tests = {
        2: "[0, 5]",
        7: "(5, 10]",
        12: "(10, 15]",
        17: "(15, 20]",
        -1: None,
        0: "[0, 5]",
        20: "(15, 20]",
        21: None,
        3.0: "[0, 5]",
        20.1: None,
        None: None,
    }
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_binning_left() -> None:
    """BinningSpec with right=False works as expected."""
    spec = BinningSpec([0, 5, 10, 15, 20], right=False)
    assert spec.bins() == ["[0, 5)", "[5, 10)", "[10, 15)", "[15, 20]"]
    bin_tests = {
        2: "[0, 5)",
        7: "[5, 10)",
        12: "[10, 15)",
        17: "[15, 20]",
        -1: None,
        0: "[0, 5)",
        20: "[15, 20]",
        21: None,
        None: None,
    }
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_binning_noninclusive() -> None:
    """BinningSpec with include_both_endpoints=False works as expected."""
    spec = BinningSpec([0, 5, 10, 15, 20], include_both_endpoints=False)
    assert spec.bins() == ["(0, 5]", "(5, 10]", "(10, 15]", "(15, 20]"]
    assert spec(0) is None
    assert spec(20) == "(15, 20]"
    spec = BinningSpec([0, 5, 10, 15, 20], right=False, include_both_endpoints=False)
    assert spec.bins() == ["[0, 5)", "[5, 10)", "[10, 15)", "[15, 20)"]
    assert spec(0) == "[0, 5)"
    assert spec(20) is None


def test_binning_custom_names() -> None:
    """BinningSpec with custom bin names works as expected."""
    spec = BinningSpec([0, 64, 69, 79, 89, 100], names=["F", "D", "C", "B", "A"])
    assert spec.bins() == ["F", "D", "C", "B", "A"]
    assert spec.column_descriptor.column_type == ColumnType.VARCHAR
    bin_tests = {0: "F", 10: "F", 75: "C", 100: "A", None: None}
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_binning_repeated_names() -> None:
    """BinningSpec with non-unique bin names works as expected."""
    spec = BinningSpec([-15, -5, 5, 15], names=["high", "low", "high"])
    assert spec.bins() == ["high", "low"]
    assert spec.bins(include_null=True) == ["high", "low", None]
    assert spec.column_descriptor.column_type == ColumnType.VARCHAR
    bin_tests = {-16: None, -15: "high", -5: "high", -4: "low", 4: "low", 10: "high"}
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_binning_inf_nan() -> None:
    """Binning infinite/NaN values works as expected."""
    spec = BinningSpec(
        [float("-inf"), 0.0, float("inf")],
        right=False,
        names=["negative", "nonnegative"],
    )
    assert spec.bins() == ["negative", "nonnegative"]
    assert spec.column_descriptor.column_type == ColumnType.VARCHAR
    bin_tests = {
        -1.0: "negative",
        0.0: "nonnegative",
        1.0: "nonnegative",
        float("-inf"): "negative",
        float("inf"): "nonnegative",
        float("nan"): None,
        None: None,
    }
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_binning_nan_bin() -> None:
    """Binning with the nan_bin option works as expected."""
    spec = BinningSpec(
        [float("-inf"), 0.0, float("inf")],
        right=False,
        names=["negative", "nonnegative"],
        nan_bin="NaN",
    )
    assert spec.bins() == ["negative", "nonnegative", "NaN"]
    assert spec.column_descriptor.column_type == ColumnType.VARCHAR
    bin_tests = {
        1.0: "nonnegative",
        float("inf"): "nonnegative",
        float("nan"): "NaN",
        None: None,
    }
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_binning_nan_bin_matching() -> None:
    """Binning with nan_bin works when the given bin matches another bin."""
    spec = BinningSpec(
        [float("-inf"), 0.0, float("inf")],
        right=False,
        names=["negative", "nonnegative"],
        nan_bin="nonnegative",
    )
    assert spec.bins() == ["negative", "nonnegative"]
    assert spec.column_descriptor.column_type == ColumnType.VARCHAR
    bin_tests = {
        1.0: "nonnegative",
        float("inf"): "nonnegative",
        float("nan"): "nonnegative",
        None: None,
    }
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_binning_date_names() -> None:
    """BinningSpecs with dates as bin names work as expected."""
    spec = BinningSpec(
        [datetime.datetime(2022, 1, day) for day in range(1, 10)],
        names=[datetime.date(2022, 1, day) for day in range(1, 9)],
        right=False,
    )
    assert spec.column_descriptor.column_type == ColumnType.DATE
    bin_tests = {
        datetime.datetime.fromisoformat("2022-01-02"): datetime.date(2022, 1, 2),
        datetime.datetime.fromisoformat("2022-01-02 00:00"): datetime.date(2022, 1, 2),
        datetime.datetime.fromisoformat("2022-01-03 05:30"): datetime.date(2022, 1, 3),
    }
    for val, expected_bin in bin_tests.items():
        assert spec(val) == expected_bin


def test_not_enough_bins() -> None:
    """Edge lists that result in zero bins are rejected."""
    with pytest.raises(ValueError, match="At least two bin edges must be provided"):
        BinningSpec([])
    with pytest.raises(ValueError, match="At least two bin edges must be provided"):
        BinningSpec([1])


@pytest.mark.parametrize(
    "edges",
    [
        ([1, 2, 3, 5, 4]),
        ([1, 2, 3, 4, 4]),
        ([1, 2, 3, 3, 4]),
        ([1, 1, 2, 3, 4]),
        ([1.0, 1.1, 1.10, 2.0]),
        ([datetime.date(2022, 1, 1), datetime.date(2022, 1, 1)]),
    ],
)
def test_unsorted_edges(edges: List[Any]):
    """Edge lists that are not sorted or contain duplicate values are rejected."""
    with pytest.raises(
        ValueError,
        match="Bin edges must be sorted in ascending order, with no duplicates",
    ):
        BinningSpec(edges)


@pytest.mark.parametrize(
    "edges",
    [
        ([1.0, 2]),
        ([1, 2, 3, 4, 5, 6.0, 7, 8]),
        ([1, "2"]),
        (["1", 2]),
        ([datetime.date(1, 1, 1), datetime.datetime(1, 1, 2)]),
        ([datetime.datetime(1, 1, 1), datetime.date(1, 1, 2)]),
    ],
)
def test_mixed_type_edges(edges: List[Any]):
    """Edge lists with non-uniform type are rejected."""
    with pytest.raises(
        ValueError, match="Invalid bin edges: list contains elements of multiple types"
    ):
        BinningSpec(edges)


@pytest.mark.parametrize("edges", [([1, None]), ([None, 1])])
def test_none_type_edges(edges: List[Any]):
    """Edge lists with non-uniform type are rejected."""
    with pytest.raises(ValueError, match="Invalid bin edges: None is not allowed"):
        BinningSpec(edges)


@pytest.mark.parametrize(
    "names",
    [
        [1.0, 2],
        [1, 2, 3, 4, 5, 6.0, 7, 8],
        [1, "2"],
        ["1", 2],
        [datetime.date(1, 1, 1), datetime.datetime(1, 1, 2)],
        [datetime.datetime(1, 1, 1), datetime.date(1, 1, 2)],
    ],
)
def test_mixed_type_names(names: List[Any]):
    """Bin name lists with non-uniform type are rejected."""
    with pytest.raises(
        ValueError, match="Invalid bin names: list contains elements of multiple types"
    ):
        BinningSpec(range(len(names) + 1), names=names)


@pytest.mark.parametrize("names", [([1, None]), ([None, 1])])
def test_none_type_names(names: List[Any]):
    """Bin name lists with non-uniform type are rejected."""
    with pytest.raises(ValueError, match="Invalid bin names: None is not allowed"):
        BinningSpec(range(len(names) + 1), names=names)


@pytest.mark.parametrize(
    "names,nan_bin", [(["0", "1"], 0), ([0, 1], "nan"), ([0.5, 1.0], "nan")]
)
def test_mismatched_nan_bin_name(names: List[Any], nan_bin: Any):
    """NaN bin names that don't match other bin names' type are rejected."""
    with pytest.raises(
        ValueError, match="NaN bin name must have the same type as other bin names"
    ):
        BinningSpec(range(len(names) + 1), names=names, nan_bin=nan_bin)


@pytest.mark.parametrize(
    "edges,names",
    [
        ([1, 2], []),
        ([1, 2], ["a", "b"]),
        ([1, 2], ["a", "b", "c"]),
        (["a", "b", "c", "d"], [1, 2, 3, 4]),
        (["a", "b", "c", "d"], [1, 2]),
    ],
)
def test_wrong_names_length(edges: List[Any], names: List[Any]):
    """Bin name lists of the wrong length are rejected."""
    with pytest.raises(
        ValueError,
        match="Number of bin names must be one less than the number of bin edges",
    ):
        BinningSpec(edges, names=names)
