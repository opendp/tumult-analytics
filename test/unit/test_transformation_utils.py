"""Unit tests for transofrmation utils."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import DictMetric, SymmetricDifference
from tmlt.core.transformations.identity import Identity as IdentityTransformation

from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics._transformation_utils import (
    delete_table,
    persist_table,
    rename_table,
    unpersist_table,
)


def test_rename_table():
    """Test rename table."""
    sdf_input_dict_domain = DictDomain(
        {
            NamedTable("private"): SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkIntegerColumnDescriptor(),
                    "X": SparkFloatColumnDescriptor(),
                }
            )
        }
    )
    identity_transformation = IdentityTransformation(
        metric=DictMetric({NamedTable("private"): SymmetricDifference()}),
        domain=sdf_input_dict_domain,
    )

    output_transformation, new_ref = rename_table(
        identity_transformation,
        TableReference([NamedTable("private")]),
        NamedTable("renamed_private"),
    )
    assert TableReference([NamedTable("renamed_private")]) == new_ref
    assert isinstance(output_transformation.output_domain, DictDomain)
    assert (
        NamedTable(name="renamed_private")
        in output_transformation.output_domain.key_to_domain.keys()
    )
    assert (
        output_transformation.output_domain[NamedTable(name="renamed_private")]
        == sdf_input_dict_domain[NamedTable("private")]
    )


def test_delete_table():
    """Test delete table."""
    sdf_input_dict_domain = DictDomain(
        {
            NamedTable("private1"): SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkIntegerColumnDescriptor(),
                    "X": SparkFloatColumnDescriptor(),
                }
            ),
            NamedTable("private2"): SparkDataFrameDomain(
                {"A": SparkStringColumnDescriptor()}
            ),
        }
    )
    identity_transformation = IdentityTransformation(
        metric=DictMetric(
            {
                NamedTable("private1"): SymmetricDifference(),
                NamedTable("private2"): SymmetricDifference(),
            }
        ),
        domain=sdf_input_dict_domain,
    )

    output_transformation = delete_table(
        identity_transformation, TableReference([NamedTable("private1")])
    )
    assert isinstance(output_transformation.output_domain, DictDomain)
    assert (
        NamedTable(name="private1")
        not in output_transformation.output_domain.key_to_domain.keys()
    )
    assert (
        NamedTable(name="private2")
        in output_transformation.output_domain.key_to_domain.keys()
    )


def test_persist_table():
    """Test persist table."""
    sdf_input_dict_domain = DictDomain(
        {
            NamedTable("private"): SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkIntegerColumnDescriptor(),
                    "X": SparkFloatColumnDescriptor(),
                }
            )
        }
    )
    identity_transformation = IdentityTransformation(
        metric=DictMetric({NamedTable("private"): SymmetricDifference()}),
        domain=sdf_input_dict_domain,
    )

    output_transformation, new_ref = persist_table(
        identity_transformation,
        TableReference([NamedTable("private")]),
        NamedTable("persisted_table"),
    )
    assert TableReference([NamedTable("persisted_table")]) == new_ref
    assert isinstance(output_transformation.output_domain, DictDomain)
    assert (
        NamedTable(name="persisted_table")
        in output_transformation.output_domain.key_to_domain.keys()
    )


def test_unpersist_table():
    """Test unpersist table."""
    sdf_input_dict_domain = DictDomain(
        {
            NamedTable("private1"): SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkIntegerColumnDescriptor(),
                    "X": SparkFloatColumnDescriptor(),
                }
            ),
            NamedTable("private2"): SparkDataFrameDomain(
                {"A": SparkStringColumnDescriptor()}
            ),
        }
    )
    identity_transformation = IdentityTransformation(
        metric=DictMetric(
            {
                NamedTable("private1"): SymmetricDifference(),
                NamedTable("private2"): SymmetricDifference(),
            }
        ),
        domain=sdf_input_dict_domain,
    )

    output_transformation = unpersist_table(
        identity_transformation, TableReference([NamedTable("private1")])
    )
    assert isinstance(output_transformation.output_domain, DictDomain)
    assert all(
        id in output_transformation.output_domain.key_to_domain.keys()
        for id in (NamedTable(name="private1"), NamedTable(name="private2"))
    )
