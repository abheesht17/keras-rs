import jax
import numpy as np
import tensorflow as tf


def _get_dummy_batch(batch_size, multi_hot_sizes, vocabulary_sizes):
    """Returns a dummy batch of data in the final desired structure."""

    # Labels.
    data = {
        "clicked": np.random.randint(0, 2, size=(batch_size,), dtype=np.int64)
    }

    # Dense features.
    dense_features_list = [
        np.random.uniform(0.0, 0.9, size=(batch_size, 1)).astype(np.float32)
        for _ in range(13)
    ]
    data["dense_features"] = np.concatenate(dense_features_list, axis=-1)

    # Sparse features.
    sparse_features = {}
    for i, (multi_hot_size, vocabulary_size) in enumerate(
        zip(multi_hot_sizes, vocabulary_sizes)
    ):
        # TODO: We don't need this custom renaming. Remove later, when we
        # shift from dummy data to actual data.
        sparse_features[f"cat_{i + 14}_id"] = np.random.randint(
            low=0,
            high=vocabulary_size,
            size=(batch_size, multi_hot_size),
            dtype=np.int64,
        )
    data["sparse_features"] = sparse_features
    return data


def create_dummy_dataset(
    batch_size, multi_hot_sizes, vocabulary_sizes, sparse_feature_preprocessor
):
    """Creates a TF dataset from cached dummy data of the final batch size."""
    dummy_data = _get_dummy_batch(batch_size, multi_hot_sizes, vocabulary_sizes)

    dataset = tf.data.Dataset.from_tensors(dummy_data).repeat(5).shard(
        jax.process_count(), jax.process_index()
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    dataset = dataset.with_options(options)

    def generator():
        for example in dataset:
            yield (
                {
                    "dense_features": example["dense_features"],
                    "preprocessed_sparse_features": sparse_feature_preprocessor.preprocess(
                        example["sparse_features"]
                    ),
                },
                example["clicked"],
            )

    return generator


# TODO: Write correct data loading logic once we have access to the dataset.

# def get_feature_spec(batch_size, dense_features, sparse_features, label):
#     feature_spec = {
#         label: tf.io.FixedLenFeature(
#             [batch_size],
#             dtype=tf.int64,
#         )
#     }

#     for dense_feature in dense_features:
#         feature_spec[dense_feature] = tf.io.FixedLenFeature(
#             [batch_size],
#             dtype=tf.float32,
#         )

#     for sparse_feature in sparse_features:
#         feature_spec[sparse_feature] = tf.io.FixedLenFeature(
#             [batch_size],
#             dtype=tf.string,
#         )

#     return feature_spec


# def preprocess(
#     example,
#     sparse_feature_preprocessor,
#     batch_size,
#     dense_features,
#     sparse_features,
#     multi_hot_sizes,
#     label,
# ):
#     # Read example.
#     feature_spec = get_feature_spec(
#         batch_size, dense_features, sparse_features, label
#     )
#     example = tf.io.parse_single_example(example, feature_spec)

#     # Dense features
#     dense_feature_list = [
#         tf.reshape(example[dense_feature], [batch_size, 1])
#         for dense_feature in dense_features
#     ]
#     dense_features = tf.stack(dense_feature_list, axis=-1)

#     # Sparse features
#     sparse_features_dict = {}
#     for sparse_feature, multi_hot_size in zip(sparse_features, multi_hot_sizes):
#         raw_values = tf.io.decode_raw(example[sparse_feature], tf.int64)
#         raw_values = tf.reshape(raw_values, [batch_size, multi_hot_size])
#         sparse_features_dict[sparse_feature] = raw_values

#     # Labels
#     labels = tf.reshape(example[label], [batch_size])

#     return (
#         {
#             "dense_features": dense_features,
#             "preprocessed_sparse_features": sparse_feature_preprocessor.preprocess(
#                 sparse_features_dict
#             ),
#         },
#         labels,
#     )


# def create_dataset(
#     file_pattern,
#     sparse_feature_preprocessor,
#     per_replica_batch_size,
#     dense_features,
#     sparse_features,
#     multi_hot_sizes,
#     vocabulary_sizes,
#     label,
#     num_processes,
#     process_id,
#     parallelism,
#     shuffle_buffer=256,
#     prefetch_size=256,
#     training=False,
#     return_dummy_dataset=False,
# ):
#     if return_dummy_dataset:
#         return _get_direct_dummy_dataset(
#             per_replica_batch_size * num_processes,
#             multi_hot_sizes,
#             vocabulary_sizes,
#             sparse_feature_preprocessor,
#         )

#     dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
#     dataset = dataset.shard(num_processes, process_id)
#     dataset = tf.data.TFRecordDataset(
#         dataset,
#         buffer_size=32 * 1024 * 1024,
#         num_parallel_reads=parallelism,
#     )

#     # Process example.
#     dataset = dataset.map(
#         lambda x: preprocess(
#             x,
#             sparse_feature_preprocessor,
#             per_replica_batch_size,
#             dense_features,
#             sparse_features,
#             multi_hot_sizes,
#             label,
#         ),
#         num_parallel_calls=parallelism,
#     )

#     # Shuffle dataset if in training mode.
#     if training and shuffle_buffer > 0:
#         dataset = dataset.shuffle(shuffle_buffer)

#     dataset = dataset.prefetch(prefetch_size)
#     options = tf.data.Options()
#     options.deterministic = False
#     options.threading.private_threadpool_size = 96
#     dataset = dataset.with_options(options)

#     return dataset
