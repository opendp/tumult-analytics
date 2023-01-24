"""Useful functions to be used with transfomations."""
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from typing import Callable, Dict, Tuple, Type

from tmlt.analytics._table_identifier import Identifier, TemporaryTable
from tmlt.analytics._table_reference import TableReference, lookup_domain, lookup_metric
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.metrics import AddRemoveKeys, DictMetric, Metric
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.dictionary import GetValue as GetValueTransformation
from tmlt.core.transformations.dictionary import create_transform_value


def generate_nested_transformation(
    base_transformation: Transformation,
    parent_reference: TableReference,
    generator_dict: Dict[
        Type[Metric], Callable[[Domain, Metric, Identifier], Transformation]
    ],
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
    """
    parent_domain = lookup_domain(base_transformation.output_domain, parent_reference)
    parent_metric = lookup_metric(base_transformation.output_metric, parent_reference)

    target_table = TemporaryTable()

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


def get_table_from_ref(
    transformation: Transformation, ref: TableReference
) -> Transformation:
    """Returns a GetValue transformation, finding the referenced table."""
    for p in ref.path:
        domain = transformation.output_domain
        metric = transformation.output_metric
        if not isinstance(domain, DictDomain):
            raise ValueError(
                "Invalid transformation domain. This is probably a bug; please let us"
                " know about it so we can fix it!"
            )
        if not isinstance(metric, (DictMetric, AddRemoveKeys)):
            raise ValueError(
                "Invalid transformation domain. This is probably a bug; please let us"
                " know about it so we can fix it!"
            )

        transformation = transformation | GetValueTransformation(domain, metric, p)
    return transformation
