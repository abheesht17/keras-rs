import numpy as np
import tensorflow as tf


def _get_dummy_batch(batch_size, large_emb_features, small_emb_features):
    """Returns a dummy batch of data in the final desired structure."""

    # Labels
    data = {
        "clicked": np.random.randint(0, 2, size=(batch_size,), dtype=np.int64)
    }

    # Dense features
    dense_input_list = [
        np.random.uniform(0.0, 0.9, size=(batch_size, 1)).astype(np.float32)
        for _ in range(13)
    ]
    data["dense_input"] = np.concatenate(dense_input_list, axis=-1)

    # Sparse features
    large_emb_inputs = {}
    for large_emb_feature in large_emb_features:
        vocabulary_size = large_emb_feature["vocabulary_size"]
        multi_hot_size = large_emb_feature["multi_hot_size"]
        idx = large_emb_feature["name"].split("-")[-1]

        large_emb_inputs[f"cat_{idx}_id"] = np.random.randint(
            low=0,
            high=vocabulary_size,
            size=(batch_size, multi_hot_size),
            dtype=np.int64,
        )

    data["large_emb_inputs"] = large_emb_inputs

    # Dense lookup features
    small_emb_inputs = {}
    for small_emb_feature in small_emb_features:
        vocabulary_size = small_emb_feature["vocabulary_size"]
        multi_hot_size = small_emb_feature["multi_hot_size"]
        idx = large_emb_feature["name"].split("-")[-1]

        # TODO: We don't need this custom renaming. Remove later, when we
        # shift from dummy data to actual data.
        small_emb_inputs[f"cat_{idx}_id"] = np.random.randint(
            low=0,
            high=vocabulary_size,
            size=(batch_size, multi_hot_size),
            dtype=np.int64,
        )

    if small_emb_inputs:
        data["small_emb_inputs"] = small_emb_inputs

    return data


def create_dummy_dataset(batch_size, large_emb_features, small_emb_features):
    """Creates a TF dataset from cached dummy data of the final batch size."""
    dummy_data = _get_dummy_batch(
        batch_size, large_emb_features, small_emb_features
    )

    # Separate labels from features to create a `(features, labels)` tuple.
    labels = dummy_data.pop("clicked")
    features = dummy_data

    dataset = tf.data.Dataset.from_tensors((features, labels)).repeat(16)
    return dataset


def get_feature_spec(batch_size, dense_features, large_emb_features, label):
    feature_spec = {
        label: tf.io.FixedLenFeature(
            [batch_size],
            dtype=tf.int64,
        )
    }

    for dense_feature in dense_features:
        feature_spec[dense_feature] = tf.io.FixedLenFeature(
            [batch_size],
            dtype=tf.float32,
        )

    for large_emb_feature in large_emb_features:
        feature_spec[large_emb_feature] = tf.io.FixedLenFeature(
            [batch_size],
            dtype=tf.string,
        )

    return feature_spec


# def preprocess(
#     example,
#     large_emb_feature_preprocessor,
#     batch_size,
#     dense_features,
#     large_emb_features,
#     multi_hot_sizes,
#     label,
# ):
#     # Read example.
#     feature_spec = get_feature_spec(
#         batch_size, dense_features, large_emb_features, label
#     )
#     example = tf.io.parse_single_example(example, feature_spec)

#     # Dense features
#     dense_feature_list = [
#         tf.reshape(example[dense_feature], [batch_size, 1])
#         for dense_feature in dense_features
#     ]
#     dense_features = tf.stack(dense_feature_list, axis=-1)

#     # Sparse features
#     large_emb_features_dict = {}
#     for large_emb_feature, multi_hot_size in zip(large_emb_features, multi_hot_sizes):
#         raw_values = tf.io.decode_raw(example[large_emb_feature], tf.int64)
#         raw_values = tf.reshape(raw_values, [batch_size, multi_hot_size])
#         large_emb_features_dict[large_emb_feature] = raw_values

#     # Labels
#     labels = tf.reshape(example[label], [batch_size])

#     return (
#         {
#             "dense_features": dense_features,
#             "preprocessed_large_emb_features": large_emb_feature_preprocessor.preprocess(
#                 large_emb_features_dict
#             ),
#         },
#         labels,
#     )


# def create_dataset(
#     file_pattern,
#     large_emb_feature_preprocessor,
#     per_replica_batch_size,
#     dense_features,
#     large_emb_features,
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
#             large_emb_feature_preprocessor,
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
#             large_emb_feature_preprocessor,
#             per_replica_batch_size,
#             dense_features,
#             large_emb_features,
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
