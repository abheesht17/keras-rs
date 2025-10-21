"""Utility functions for manipulating JAX embedding tables and inputs."""

import collections
import dataclasses
import typing
from typing import Any, Mapping, NamedTuple, Sequence, TypeAlias, TypeVar

import jax
import numpy as np
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import FeatureSpec
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import StackedTableSpec
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import TableSpec

from keras_rs.src.types import Nested

T = TypeVar("T")

# Any to support tf.Ragged without needing an explicit TF dependency.
ArrayLike: TypeAlias = jax.Array | np.ndarray | Any  # type: ignore
Shape: TypeAlias = tuple[int, ...]


class FeatureSamples(NamedTuple):
    tokens: ArrayLike
    weights: ArrayLike


class ShardedCooMatrix(NamedTuple):
    shard_starts: ArrayLike
    shard_ends: ArrayLike
    col_ids: ArrayLike
    row_ids: ArrayLike
    values: ArrayLike


class InputStatsPerTable(NamedTuple):
    max_ids_per_partition: int
    max_unique_ids_per_partition: int
    required_buffer_size_per_device: int


def convert_to_numpy(
    ragged_or_dense: np.ndarray[Any, Any] | Sequence[Sequence[Any]] | Any,
    dtype: Any,
) -> np.ndarray[Any, Any]:
    """Converts a ragged or dense list of inputs to a ragged/dense numpy array.

    The output is adjusted to be 2D.

    Args:
        ragged_or_dense: Input that is either already a numpy array, or nested
            sequence.
        dtype: Numpy dtype of output array.

    Returns:
        Corresponding numpy array.
    """
    if hasattr(ragged_or_dense, "numpy"):
        # Support tf.RaggedTensor and other TF input dtypes.
        if callable(getattr(ragged_or_dense, "numpy")):
            ragged_or_dense = ragged_or_dense.numpy()

    if isinstance(ragged_or_dense, jax.Array):
        ragged_or_dense = np.asarray(ragged_or_dense)

    if isinstance(ragged_or_dense, np.ndarray):
        # Convert 1D to 2D.
        if ragged_or_dense.dtype != np.ndarray and ragged_or_dense.ndim == 1:
            return ragged_or_dense.reshape(-1, 1).astype(dtype)

        # If dense, return converted dense type.
        if ragged_or_dense.dtype != np.ndarray:
            return ragged_or_dense.astype(dtype)

        # Ragged numpy array.
        return ragged_or_dense

    # Handle 1D sequence input.
    if not isinstance(ragged_or_dense[0], collections.abc.Sequence):
        return np.asarray(ragged_or_dense, dtype=dtype).reshape(-1, 1)

    # Assemble elements into an nd-array.
    counts = [len(vals) for vals in ragged_or_dense]
    if all([count == counts[0] for count in counts]):
        # Dense input.
        return np.asarray(ragged_or_dense, dtype=dtype)
    else:
        # Ragged input, convert to ragged numpy arrays.
        return np.array(
            [np.array(row, dtype=dtype) for row in ragged_or_dense],
            dtype=np.ndarray,
        )


def ones_like(
    ragged_or_dense: np.ndarray[Any, Any], dtype: Any = None
) -> np.ndarray[Any, Any]:
    """Creates an array of ones the same as as the input.

    This differs from traditional numpy in that a ragged input will lead to
    a resulting ragged array of ones, whereas np.ones_like(...) will instead
    only consider the outer array and return a 1D dense array of ones.

    Args:
        ragged_or_dense: The ragged or dense input whose shape and data-type
            define these same attributes of the returned array.
        dtype: The data-type of the returned array.

    Returns:
        An array of ones with the same shape as the input, and specified data
        type.
    """
    dtype = dtype or ragged_or_dense.dtype
    if ragged_or_dense.dtype == np.ndarray:
        # Ragged.
        return np.array(
            [np.ones_like(row, dtype=dtype) for row in ragged_or_dense],
            dtype=np.ndarray,
        )
    else:
        # Dense.
        return np.ones_like(ragged_or_dense, dtype=dtype)


def create_feature_samples(
    feature_structure: Nested[T],
    feature_ids: Nested[ArrayLike | Sequence[int] | Sequence[Sequence[int]]],
    feature_weights: None
    | (Nested[ArrayLike | Sequence[float] | Sequence[Sequence[float]]]),
) -> Nested[FeatureSamples]:
    """Constructs a collection of sample tuples from provided IDs and weights.

    Args:
        feature_structure: The nested structure of the inputs (typically
            `FeatureSpec`s).
        feature_ids: The feature IDs to use for the samples.
        feature_weights: The feature weights to use for the samples.  Defaults
            to ones if not provided.

    Returns:
        A nested collection of `FeatureSamples` corresponding to the input IDs
        and weights, for use in embedding lookups.
    """
    # Create numpy arrays from inputs.
    feature_ids = jax.tree.map(
        lambda _, ids: convert_to_numpy(ids, np.int32),
        feature_structure,
        feature_ids,
    )

    if feature_weights is None:
        # Make ragged or dense ones_like.
        feature_weights = jax.tree.map(
            lambda _, ids: ones_like(ids, np.float32),
            feature_structure,
            feature_ids,
        )
    else:
        feature_weights = jax.tree.map(
            lambda _, wgts: convert_to_numpy(wgts, np.float32),
            feature_structure,
            feature_weights,
        )

    # Assemble.
    def _create_feature_samples(
        sample_ids: np.ndarray[Any, Any],
        sample_weights: np.ndarray[Any, Any],
    ) -> FeatureSamples:
        return FeatureSamples(sample_ids, sample_weights)

    output: Nested[FeatureSamples] = jax.tree.map(
        lambda _, sample_ids, sample_weights: _create_feature_samples(
            sample_ids, sample_weights
        ),
        feature_structure,
        feature_ids,
        feature_weights,
    )
    return output


def stack_and_shard_samples(
    feature_specs: Nested[FeatureSpec],
    feature_samples: Nested[FeatureSamples],
    local_device_count: int,
    global_device_count: int,
    num_sc_per_device: int,
    static_buffer_size: int | Mapping[str, int] | None = None,
) -> tuple[dict[str, ShardedCooMatrix], embedding.SparseDenseMatmulInputStats]:
    """Prepares input samples for use in embedding lookups.

    Args:
        feature_specs: Nested collection of feature specifications.
        feature_samples: Nested collection of feature samples.
        local_device_count: Number of local JAX devices.
        global_device_count: Number of global JAX devices.
        num_sc_per_device: Number of sparsecores per device.
        static_buffer_size: The static buffer size to use for the samples.
          Defaults to None, in which case an upper-bound for the buffer size
          will be automatically determined.

    Returns:
        The preprocessed inputs, and statistics useful for updating FeatureSpecs
        based on the provided input data.
    """
    del static_buffer_size  # Currently ignored.
    flat_feature_specs, _ = jax.tree.flatten(feature_specs)

    feature_tokens = []
    feature_weights = []

    def collect_tokens_and_weights(
        feature_spec: FeatureSpec, samples: FeatureSamples
    ) -> None:
        del feature_spec
        feature_tokens.append(samples.tokens)
        feature_weights.append(samples.weights)

    jax.tree.map(collect_tokens_and_weights, feature_specs, feature_samples)

    preprocessed_inputs, stats = embedding.preprocess_sparse_dense_matmul_input(
        feature_tokens,
        feature_weights,
        flat_feature_specs,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        has_leading_dimension=False,
        allow_id_dropping=True,
    )

    out: dict[str, ShardedCooMatrix] = {}
    tables_names = preprocessed_inputs.lhs_row_pointers.keys()
    for table_name in tables_names:
        shard_ends = preprocessed_inputs.lhs_row_pointers[table_name]
        shard_starts = np.concatenate(
            [
                np.asarray([0]),
                table_stacking._next_largest_multiple(shard_ends[:-1], 8),
            ]
        )
        out[table_name] = ShardedCooMatrix(
            shard_starts=shard_starts,
            shard_ends=shard_ends,
            col_ids=preprocessed_inputs.lhs_embedding_ids[table_name],
            row_ids=preprocessed_inputs.lhs_sample_ids[table_name],
            values=preprocessed_inputs.lhs_gains[table_name],
        )

    return out, stats


def get_stacked_table_stats(
    feature_specs: Nested[FeatureSpec],
) -> dict[str, InputStatsPerTable]:
    """Extracts the stacked-table input statistics from the feature specs.

    Args:
        feature_specs: Feature specs from which to extracts the statistics.

    Returns:
        A mapping of stacked table names to input statistics per table.
    """
    stacked_table_specs: dict[str, StackedTableSpec] = {}
    for feature_spec in jax.tree.flatten(feature_specs)[0]:
        feature_spec = typing.cast(FeatureSpec, feature_spec)
        stacked_table_spec = typing.cast(
            StackedTableSpec, feature_spec.table_spec.stacked_table_spec
        )
        stacked_table_specs[stacked_table_spec.stack_name] = stacked_table_spec

    stats: dict[str, InputStatsPerTable] = {}
    for stacked_table_spec in stacked_table_specs.values():
        buffer_size = stacked_table_spec.suggested_coo_buffer_size_per_device
        buffer_size = buffer_size or 0
        stats[stacked_table_spec.stack_name] = InputStatsPerTable(
            max_ids_per_partition=stacked_table_spec.max_ids_per_partition,
            max_unique_ids_per_partition=stacked_table_spec.max_unique_ids_per_partition,
            required_buffer_size_per_device=buffer_size,
        )

    return stats


def update_stacked_table_stats(
    feature_specs: Nested[FeatureSpec],
    stats: Mapping[str, InputStatsPerTable],
) -> None:
    """Updates stacked-table input properties in the supplied feature specs.

    Args:
        feature_specs: Feature specs to update in-place.
        stats: Per-stacked-table input statistics.
    """
    # Collect table specs and stacked table specs.
    table_specs: dict[str, TableSpec] = {}
    for feature_spec in jax.tree.flatten(feature_specs)[0]:
        feature_spec = typing.cast(FeatureSpec, feature_spec)
        table_specs[feature_spec.table_spec.name] = feature_spec.table_spec

    stacked_table_specs: dict[str, StackedTableSpec] = {}
    for table_spec in table_specs.values():
        stacked_table_spec = typing.cast(
            StackedTableSpec, table_spec.stacked_table_spec
        )
        stacked_table_specs[stacked_table_spec.stack_name] = stacked_table_spec

    # Replace fields in the stacked_table_specs.
    stack_names = stacked_table_specs.keys()
    for stack_name in stack_names:
        stack_stats = stats[stack_name]
        stacked_table_spec = stacked_table_specs[stack_name]
        buffer_size = stack_stats.required_buffer_size_per_device or None
        stacked_table_specs[stack_name] = dataclasses.replace(
            stacked_table_spec,
            max_ids_per_partition=stack_stats.max_ids_per_partition,
            max_unique_ids_per_partition=stack_stats.max_unique_ids_per_partition,
            suggested_coo_buffer_size_per_device=buffer_size,
        )

    # Insert new stacked tables into tables.
    for table_spec in table_specs.values():
        stacked_table_spec = typing.cast(
            StackedTableSpec, table_spec.stacked_table_spec
        )
        table_spec.stacked_table_spec = stacked_table_specs[
            stacked_table_spec.stack_name
        ]
