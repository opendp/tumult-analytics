"""A BinningSpec defines a binning operation on a column."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import datetime
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Sequence, Tuple, TypeVar, Union, cast

from tmlt.core.utils.type_utils import get_element_type

from tmlt.analytics._schema import ColumnDescriptor, ColumnType, column_type_to_py_type

BinT = TypeVar("BinT", str, Union[int, float], datetime.date, datetime.datetime)
BinNameT = TypeVar("BinNameT", str, int, float, datetime.date, datetime.datetime)


def _get_column_descriptor(
    bin_names: Sequence[Any], nan_bin: Optional[BinNameT]
) -> ColumnDescriptor:
    """Return the ColumnDescriptor for the non-``None`` elements of a list.

    * allow_nan is True if
       - any bin name, including nan_bin, is NaN.
    * allow_inf is True if
       - any bin name, including nan_bin, matches float("inf") or float("-inf").
    * allow_null is True if (for simplicity, we always set it to be True)
           - A value is Null.
           - A value is out of bounds.
           - A value is NaN, and nan_bin is not used.
    """
    if nan_bin is not None:
        all_bin_names = list(bin_names) + [nan_bin]
    else:
        all_bin_names = list(bin_names)
    allow_nan = any(
        isinstance(bin_name, float) and math.isnan(bin_name)
        for bin_name in all_bin_names
    )
    allow_inf = any(
        bin_name in [float("inf"), float("-inf")] for bin_name in all_bin_names
    )
    allow_null = True

    # nan_bin type is checked in the __init__ method.
    column_type = ColumnType(get_element_type(bin_names, allow_none=False))
    return ColumnDescriptor(column_type, allow_null, allow_nan, allow_inf)


def _edges_as_str(bin_edges: Tuple[BinT]) -> Tuple[str, ...]:
    """Format the bin edges as strings."""
    if isinstance(bin_edges[0], float):
        # 17 is roughly the maximum number of decimal digits that can be encoded
        # in a double, so going for more than 17 digits of precision won't help.
        for precision in range(2, 17):
            bin_edge_strs = tuple(f"{e:.{precision}f}" for e in bin_edges)
            if len(bin_edge_strs) == len(set(bin_edge_strs)):
                return bin_edge_strs
        raise RuntimeError(
            "Unable to generate distinct bin edges. This should not happen and is "
            "probably a bug; please let us know about it so we can fix it!"
        )
    if isinstance(bin_edges[0], datetime.datetime):
        # Mypy has a bug (https://github.com/python/mypy/issues/9015) where
        # datetimes and dates are considered interchangeable in places where
        # they are not, which causes mypy to think this case can be executed
        # when the bin edges are dates, even though at runtime that can't
        # happen.
        if any(e.microsecond % 1000 for e in bin_edges):  # type: ignore
            timespec = "microseconds"
        elif any(e.microsecond for e in bin_edges):  # type: ignore
            timespec = "milliseconds"
        elif any(e.second for e in bin_edges):  # type: ignore
            timespec = "seconds"
        else:
            timespec = "minutes"
        return tuple(
            e.isoformat(sep=" ", timespec=timespec) for e in bin_edges  # type: ignore
        )
    elif isinstance(bin_edges[0], str):
        # Use repr for strings so that they get quoted.
        return tuple(repr(e) for e in bin_edges)
    else:
        return tuple(str(e) for e in bin_edges)


def _default_bin_names(
    bin_edges: Tuple[BinT], right: bool, include_edges: bool
) -> List[str]:
    """Generate the default list of bin names from the list of bin edges."""
    bin_edge_strs = _edges_as_str(bin_edges)
    if right:
        if include_edges:
            return [f"[{bin_edge_strs[0]}, {bin_edge_strs[1]}]"] + [
                f"({bin_edge_strs[i]}, {bin_edge_strs[i+1]}]"
                for i in range(1, len(bin_edges) - 1)
            ]
        else:
            return [
                f"({bin_edge_strs[i]}, {bin_edge_strs[i+1]}]"
                for i in range(len(bin_edges) - 1)
            ]
    else:
        if include_edges:
            return [
                f"[{bin_edge_strs[i]}, {bin_edge_strs[i+1]})"
                for i in range(len(bin_edges) - 2)
            ] + [f"[{bin_edge_strs[-2]}, {bin_edge_strs[-1]}]"]
        else:
            return [
                f"[{bin_edge_strs[i]}, {bin_edge_strs[i+1]})"
                for i in range(len(bin_edges) - 1)
            ]


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BinningSpec(Generic[BinT, BinNameT]):
    """A spec object defining an operation where values are assigned to bins.

    A BinningSpec divides values into bins based on a list of bin edges, for use
    with the :meth:`~tmlt.analytics.query_builder.QueryBuilder.bin_column` method.
    All :data:`supported data types <tmlt.analytics.session.SUPPORTED_SPARK_TYPES>`
    can be binned using a BinningSpec.

    Values outside the range of the provided bins and ``None`` types are all
    mapped to ``None`` (``null`` in Spark), as are NaN values by default. Bin
    names are generated based on the bin edges, but custom names can be provided.

    By default, the right edge of each bin is included in that bin: using edges
    `[0, 5, 10]` will lead to bins `[0, 5]` and `(5, 10]`. To include the
    left edge instead, set the ``right`` parameter to False.

    Examples:
        >>> spec = BinningSpec([0,5,10])
        >>> spec.bins()
        ['[0, 5]', '(5, 10]']
        >>> spec(0)
        '[0, 5]'
        >>> spec(5)
        '[0, 5]'
        >>> spec(6)
        '(5, 10]'
        >>> spec(10)
        '(5, 10]'
        >>> spec(11) is None
        True
    """

    bin_edges: Tuple[BinT]
    names: Sequence[Optional[BinNameT]]
    right: bool
    include_both_endpoints: bool
    nan_bin: BinNameT
    _input_type: ColumnType
    _column_descriptor: ColumnDescriptor

    def __init__(
        self,
        bin_edges: Sequence[BinT],
        names: Optional[Sequence[Optional[BinNameT]]] = None,
        right: bool = True,
        include_both_endpoints: bool = True,
        nan_bin: Optional[BinNameT] = None,
    ):
        """Initialize a BinningSpec.

        Args:
            bin_edges: A list of the bin edges, sorted in ascending order.
            names: If given, used as the names of bins. Must be one element
                shorter than ``bin_edges``. Duplicate values are allowed, which
                will place non-contiguous ranges of values into the same
                bin. Note that while using floats and timestamps as bin names is
                allowed here, grouping on the resulting column is not allowed.
            right: When True, the right edge of each bin is included in that bin;
                otherwise, the left edge is. Defaults to True.
            include_both_endpoints: When True, the outer edges of both the first
                and last bins will be included in their respective bins; when
                False, these edges are treated the same as the other bins,
                i.e. only one will be included based on how ``right`` is
                set. Defaults to True.
            nan_bin: If binning over a float-valued column, all NaNs will be
                placed in a bin with this name. The default value, ``None``,
                causes these values to be placed in the same bin with
                out-of-range and null values.
        """
        num_bins = len(bin_edges) - 1
        if num_bins < 1:
            raise ValueError("At least two bin edges must be provided")
        try:
            input_type = get_element_type(bin_edges, allow_none=False)
        except ValueError as e:
            raise ValueError(f"Invalid bin edges: {e}") from e
        if not all(bin_edges[i] < bin_edges[i + 1] for i in range(num_bins)):
            raise ValueError(
                "Bin edges must be sorted in ascending order, with no duplicates"
            )

        # The class is frozen, so we need to subvert it to update attributes.
        object.__setattr__(self, "bin_edges", tuple(bin_edges))
        object.__setattr__(self, "_input_type", ColumnType(input_type))

        if names is None:
            # Assigning to self.names doesn't typecheck because of a
            # deficiency in mypy where making the names parameter optional
            # prevents it from inferring BinNameT correctly in the case where
            # names is None. If https://github.com/python/mypy/issues/3737 is
            # ever resolved, that should allow this to typecheck.
            new_bins_names = _default_bin_names(
                self.bin_edges, right, include_both_endpoints
            )
            object.__setattr__(self, "names", tuple(new_bins_names))
        else:
            if len(names) != num_bins:
                raise ValueError(
                    "Number of bin names must be one less than the number of bin edges"
                )
            object.__setattr__(self, "names", tuple(names))

        try:
            column_descriptor = _get_column_descriptor(self.names, nan_bin)
        except ValueError as e:
            raise ValueError(f"Invalid bin names: {e}") from e
        # This typecheck cannot be done safely with isinstance because datetime
        # is a subclass of date.
        if (
            # pylint: disable=unidiomatic-typecheck
            nan_bin is not None
            and type(nan_bin) != column_type_to_py_type(column_descriptor.column_type)
        ):
            raise ValueError("NaN bin name must have the same type as other bin names")

        object.__setattr__(self, "nan_bin", nan_bin)
        object.__setattr__(self, "_column_descriptor", column_descriptor)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "include_both_endpoints", include_both_endpoints)

    def __eq__(self, other: Any):
        """Adds equality comparison to the BinningSpec class."""
        if not isinstance(other, BinningSpec):
            raise TypeError(f"Cannot compare BinningSpec with {type(other)}")
        return (
            self.bin_edges == other.bin_edges
            and self.names == other.names
            and self.right == other.right
            and self.include_both_endpoints == other.include_both_endpoints
            and self.nan_bin == other.nan_bin
        )

    def __hash__(self):
        """Hashes the bin spec on a tuple of its attributes."""
        return hash(
            (
                self.bin_edges,
                self.names,
                self.right,
                self.include_both_endpoints,
                self.nan_bin,
            )
        )

    def __repr__(self):
        """Returns a string representation of the BinningSpec."""
        return (
            f"BinningSpec(bin_edges={list(self.bin_edges)}, "
            f"names={self.names}, right={self.right}, "
            f"include_both_endpoints={self.include_both_endpoints}, "
            f"nan_bin={self.nan_bin})"
        )

    @property
    def input_type(self) -> ColumnType:
        """Return the ColumnType of the column this binning can be applied to."""
        return self._input_type

    @property
    def column_descriptor(self) -> ColumnDescriptor:
        """Return the ColumnDescriptor that results from applying this binning."""
        return self._column_descriptor

    def bins(self, include_null: bool = False) -> List[Optional[BinNameT]]:
        """Return a list of all the bin names that could result from the binning.

        The returned list is guaranteed to contain unique elements, even if
        multiple bins were mapped to the same name. The NaN bin, if one was
        specified, is included. If ``include_null`` is true, the null bin is
        included as well; by default, it is not included.
        """
        names = cast(List[Optional[BinNameT]], list(self.names))
        if self.nan_bin is not None:
            names.append(self.nan_bin)
        if include_null:
            names.append(None)
        # This conversion is a trick to deduplicate values in `names` while
        # preserving the order in which they first appeared.
        return list(dict.fromkeys(names))

    def __call__(self, val: Optional[BinT]) -> Any:
        """Given a value to bin, return its bin name.

        In most cases this method only needs to be used internally, but it can
        be called on its own to test the binning that will be performed.

        Args:
            val: The value to be assigned to a bin.
        """
        if val is None:
            return None
        if isinstance(val, float) and math.isnan(val):
            return self.nan_bin
        # Note that "left" and "right" in the bisect methods refer to which side
        # of an equal value the value being searched for is considered to fall
        # on. This is kind of opposite to the meaning of self.right, so
        # bisect_left is used when self.right is True and vice versa.
        if self.right:
            if self.include_both_endpoints and val == self.bin_edges[0]:
                return self.names[0]
            if val <= self.bin_edges[0] or val > self.bin_edges[-1]:
                return None
            bin_position = bisect_left(self.bin_edges, val) - 1
            return self.names[bin_position]
        if self.include_both_endpoints and val == self.bin_edges[-1]:
            return self.names[-1]
        if val < self.bin_edges[0] or val >= self.bin_edges[-1]:
            return None
        bin_position = bisect_right(self.bin_edges, val) - 1
        return self.names[bin_position]
