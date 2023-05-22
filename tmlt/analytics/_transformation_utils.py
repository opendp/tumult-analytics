"""Useful functions to be used with transfomations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=unused-argument

from typing import Callable, Dict, Optional, Tuple, Type, cast

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import AddRemoveKeys, DictMetric, Metric
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.dictionary import GetValue as GetValueTransformation
from tmlt.core.transformations.dictionary import Subset as SubsetTransformation
from tmlt.core.transformations.dictionary import (
    create_copy_and_transform_value,
    create_transform_value,
)
from tmlt.core.transformations.identity import Identity
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    PersistValue as PersistValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    RenameValue as RenameValueTransformation,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    UnpersistValue as UnpersistValueTransformation,
)
from tmlt.core.transformations.spark_transformations.persist import (
    Persist as PersistTransformation,
)
from tmlt.core.transformations.spark_transformations.persist import (
    Unpersist as UnpersistTransformation,
)

from tmlt.analytics._table_identifier import Identifier, TemporaryTable
from tmlt.analytics._table_reference import TableReference, lookup_domain, lookup_metric


def generate_nested_transformation(
    base_transformation: Transformation,
    parent_reference: TableReference,
    generator_dict: Dict[
        Type[Metric], Callable[[Domain, Metric, Identifier], Transformation]
    ],
    table_identifier: Optional[Identifier] = None,
) -> Tuple[Transformation, TableReference]:
    """Generate a nested transformation.

    At a high level, this function chooses an appropriate transformation
    by keying into the generator dictionary at the parent_reference's Metric,
    then applies the selected transformation at parent_reference in the output
    of base_transformation.

    Args:
        base_transformation: Transformation to be used as a starting point.
        parent_reference: The parent TableReference of base_transformation's associated
         TableReference.
        generator_dict: a dictionary of the form { Metric : generator() },
          where generator() is a function which takes the associated domain and metric
          of the parent_reference and generates a transormation using the
          Metric, e.g. generator(parent_domain, parent_metric, target_identifier)
        table_identifier: identifier for new table in case of rename and original
          table in case of delete/persist/unpersist
    """
    parent_domain = lookup_domain(base_transformation.output_domain, parent_reference)
    parent_metric = lookup_metric(base_transformation.output_metric, parent_reference)

    target_table = TemporaryTable() if table_identifier is None else table_identifier

    try:
        gen_transformation = generator_dict[type(parent_metric)]
    except KeyError:
        raise ValueError(
            f"No matching metric for {type(parent_metric).__name__} in"
            " transformation generator."
        ) from None
    transformation = gen_transformation(parent_domain, parent_metric, target_table)

    ref = parent_reference
    while ref.path:
        identifier = ref.identifier
        ref = ref.parent

        parent_domain = lookup_domain(base_transformation.output_domain, ref)
        parent_metric = lookup_metric(base_transformation.output_metric, ref)
        if not isinstance(parent_domain, DictDomain):
            raise ValueError(
                f"The parent reference should be a {DictDomain},"
                f"but it is a {parent_domain}."
            )
        if not isinstance(parent_metric, DictMetric):
            raise ValueError(
                f"The parent reference should be a {DictMetric},"
                f"but it is a {parent_metric}."
            )
        transformation = create_transform_value(
            parent_domain, parent_metric, identifier, transformation, lambda *args: None
        )
    return base_transformation | transformation, parent_reference / target_table


def rename_table(
    base_transformation: Transformation,
    base_ref: TableReference,
    new_table_id: Identifier,
) -> Tuple[Transformation, TableReference]:
    """Renames tables.

    A single value is transformed and added to the dictionary at a new key.
    Note that the original value is left unchanged in the dictionary.
    """

    def gen_transformation_dictmetric(pd, pm, tgt):
        assert isinstance(pd, DictDomain)
        assert isinstance(pm, DictMetric)
        # Note: create_rename drops the original key, hence using this instead
        return create_copy_and_transform_value(
            input_domain=pd,
            input_metric=pm,
            key=base_ref.identifier,
            new_key=tgt,
            transformation=Identity(
                domain=pd[base_ref.identifier], metric=pm[base_ref.identifier]
            ),
            hint=lambda d_in, _: d_in,
        )

    def gen_transformation_ark(pd, pm, tgt):
        assert isinstance(pd, DictDomain)
        assert isinstance(pm, AddRemoveKeys)
        # Note: No dataframe column is getting renamed here, RenameValue
        # is used to rename tables
        return RenameValueTransformation(pd, pm, base_ref.identifier, tgt, {})

    transformation_generators: Dict[Type[Metric], Callable] = {
        DictMetric: gen_transformation_dictmetric,
        AddRemoveKeys: gen_transformation_ark,
    }
    new_transformation, new_ref = generate_nested_transformation(
        base_transformation, base_ref.parent, transformation_generators, new_table_id
    )
    return new_transformation, new_ref


def delete_table(
    base_transformation: Transformation, base_ref: TableReference
) -> Transformation:
    """Deletes tables."""

    def gen_transformation(pd, pm, tgt):
        assert isinstance(pd, DictDomain)
        assert isinstance(pm, (DictMetric, AddRemoveKeys))
        return SubsetTransformation(
            pd, pm, list(set(pd.key_to_domain.keys()) - {base_ref.identifier})
        )

    transformation_generators: Dict[Type[Metric], Callable] = {
        DictMetric: gen_transformation,
        AddRemoveKeys: gen_transformation,
    }
    new_transformation, _ = generate_nested_transformation(
        base_transformation,
        base_ref.parent,
        transformation_generators,
        base_ref.identifier,
    )
    return new_transformation


def persist_table(
    base_transformation: Transformation,
    base_ref: TableReference,
    new_table_id: Optional[Identifier] = None,
) -> Tuple[Transformation, TableReference]:
    """Persists tables."""

    def gen_transformation_dictmetric(pd, pm, tgt):
        assert isinstance(pd, DictDomain)
        assert isinstance(pm, DictMetric)
        return create_copy_and_transform_value(
            pd,
            pm,
            base_ref.identifier,
            tgt,
            PersistTransformation(
                domain=cast(
                    SparkDataFrameDomain, pd.key_to_domain[base_ref.identifier]
                ),
                metric=pm.key_to_metric[base_ref.identifier],
            ),
            lambda *args: None,
        )

    def gen_transformation_ark(pd, pm, tgt):
        assert isinstance(pd, DictDomain)
        assert isinstance(pm, AddRemoveKeys)
        return PersistValueTransformation(pd, pm, base_ref.identifier, tgt)

    transformation_generators: Dict[Type[Metric], Callable] = {
        DictMetric: gen_transformation_dictmetric,
        AddRemoveKeys: gen_transformation_ark,
    }
    new_transformation, new_ref = generate_nested_transformation(
        base_transformation, base_ref.parent, transformation_generators, new_table_id
    )
    return new_transformation, new_ref


def unpersist_table(
    base_transformation: Transformation, base_ref: TableReference
) -> Transformation:
    """Unpersists tables."""

    def gen_transformation_dictmetric(pd, pm, tgt):
        assert isinstance(pd, DictDomain)
        assert isinstance(pm, DictMetric)
        return create_copy_and_transform_value(
            pd,
            pm,
            base_ref.identifier,
            tgt,
            UnpersistTransformation(
                domain=cast(
                    SparkDataFrameDomain, pd.key_to_domain[base_ref.identifier]
                ),
                metric=pm.key_to_metric[base_ref.identifier],
            ),
            lambda *args: None,
        )

    def gen_transformation_ark(pd, pm, tgt):
        assert isinstance(pd, DictDomain)
        assert isinstance(pm, AddRemoveKeys)
        return UnpersistValueTransformation(pd, pm, base_ref.identifier, tgt)

    transformation_generators: Dict[Type[Metric], Callable] = {
        DictMetric: gen_transformation_dictmetric,
        AddRemoveKeys: gen_transformation_ark,
    }
    new_transformation, _ = generate_nested_transformation(
        base_transformation, base_ref.parent, transformation_generators
    )
    return new_transformation


def get_table_from_ref(
    transformation: Transformation, ref: TableReference
) -> Transformation:
    """Returns a GetValue transformation finding the table specified."""
    for p in ref.path:
        domain = transformation.output_domain
        metric = transformation.output_metric
        assert isinstance(domain, DictDomain), (
            "Invalid transformation domain. This is probably a bug; please let us"
            " know about it so we can fix it!"
        )
        assert isinstance(metric, (DictMetric, AddRemoveKeys)), (
            "Invalid transformation domain. This is probably a bug; please let us"
            " know about it so we can fix it!"
        )

        transformation = transformation | GetValueTransformation(domain, metric, p)
    return transformation
