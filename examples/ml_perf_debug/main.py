import sys

sys.path.append("/home/abheesht_google_com/keras-rs")


import argparse
import os

import jax

# jax.config.update("jax_debug_nans", True)
import yaml

os.environ["KERAS_BACKEND"] = "jax"


import keras

keras.config.disable_traceback_filtering()
from dataloader import create_dummy_dataset
from model import DLRMDCNV2

import keras_rs

SEED = 1337


def main(
    file_pattern,
    dense_features,
    large_emb_features,
    small_emb_features,
    label,
    embedding_dim,
    allow_id_dropping,
    max_ids_per_partition,
    max_unique_ids_per_partition,
    embedding_learning_rate,
    bottom_mlp_dims,
    top_mlp_dims,
    num_dcn_layers,
    dcn_projection_dim,
    learning_rate,
    global_batch_size,
    num_epochs,
):
    # This is to manually shard the dataset across TPUs.
    # pd = P("x")
    # global_devices = jax.devices()
    # mesh = jax.sharding.Mesh(global_devices, "x")
    # global_sharding = jax.sharding.NamedSharding(mesh, pd)

    # Set DDP as Keras distribution strategy
    distribution = keras.distribution.DataParallel()
    keras.distribution.set_distribution(distribution)

    per_host_batch_size = global_batch_size // jax.process_count()

    # === Distributed embeddings' configs for sparse features ===
    feature_configs = {}
    for large_emb_feature in large_emb_features:
        # Rename these features to something shorter; was facing some weird
        # issues with the longer names.
        feature_name = (
            large_emb_feature["name"]
            .replace("-", "_")
            .replace("egorical_feature", "")
        )
        vocabulary_size = large_emb_feature["vocabulary_size"]
        multi_hot_size = large_emb_feature["multi_hot_size"]

        table_config = keras_rs.layers.TableConfig(
            name=f"{feature_name}_table",
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            # TODO(abheesht): Verify.
            initializer="zeros",
            # keras.initializers.VarianceScaling(
            #     scale=1.0,
            #     mode="fan_in",
            #     distribution="uniform",
            #     seed=SEED,
            # ),
            optimizer=keras.optimizers.Adagrad(
                learning_rate=embedding_learning_rate
            ),
            combiner="sum",
            placement="sparsecore",
            # TODO: These two args are not getting passed down to
            # `jax-tpu-embedding` properly, seems like.
            max_ids_per_partition=max_ids_per_partition,
            max_unique_ids_per_partition=max_unique_ids_per_partition,
        )
        feature_configs[f"{feature_name}_id"] = keras_rs.layers.FeatureConfig(
            name=feature_name.replace("id", ""),
            table=table_config,
            # TODO: Verify whether it should be `(bsz, 1)` or
            # `(bsz, multi_hot_size)`.
            input_shape=(global_batch_size, multi_hot_size),
            output_shape=(global_batch_size, embedding_dim),
        )

    # === Instantiate model ===
    # We instantiate the model first, because we need to preprocess sparse
    # inputs using the distributed embedding layer defined inside the model
    # class.
    print("===== Initialising model =====")
    model = DLRMDCNV2(
        large_emb_feature_configs=feature_configs,
        small_emb_features=small_emb_features,
        embedding_dim=embedding_dim,
        bottom_mlp_dims=bottom_mlp_dims,
        top_mlp_dims=top_mlp_dims,
        num_dcn_layers=num_dcn_layers,
        dcn_projection_dim=dcn_projection_dim,
        seed=SEED,
        dtype="float32",
        name="dlrm_dcn_v2",
    )
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adagrad(learning_rate=learning_rate),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    # === Load dataset ===
    print("===== Loading dataset =====")
    train_ds = create_dummy_dataset(
        batch_size=per_host_batch_size,
        large_emb_features=large_emb_features,
        small_emb_features=small_emb_features,
    )
    # For the multi-host case, the dataset has to be distributed manually.
    # See note here:
    # https://github.com/keras-team/keras-rs/blob/main/keras_rs/src/layers/embedding/base_distributed_embedding.py#L352-L363.
    if jax.process_count() > 1:
        train_ds = distribution.distribute_dataset(train_ds)
        distribution.auto_shard_dataset = False

    # for ele in train_ds:
    #     print("--->", ele[0]["large_emb_inputs"]["cat_14_id"].shape)
    #     break

    # make_global_view = lambda x: jax.tree.map(
    #     lambda y: jax.make_array_from_process_local_data(global_sharding, y),
    #     x,
    # )

    def generator(dataset, training=False):
        """Converts tf.data Dataset to a Python generator and preprocesses
        sparse features.
        """
        for features, labels in dataset:
            # large_emb_inputs = features["large_emb_inputs"]
            # for k, v in large_emb_inputs.items():
            #     large_emb_inputs[k] = v.numpy()

            # small_emb_inputs = features["small_emb_inputs"]
            # for k, v in small_emb_inputs.items():
            #     small_emb_inputs[k] = v.numpy()

            # features["dense_input"] = features["dense_input"].numpy()

            preprocessed_large_embeddings = model.embedding_layer.preprocess(
                features["large_emb_inputs"], training=training
            )

            x = {
                "dense_input": features["dense_input"],
                "large_emb_inputs": preprocessed_large_embeddings,
                "small_emb_inputs": features["small_emb_inputs"],
            }
            y = labels
            yield (x, y)

    train_generator = generator(train_ds, training=True)
    for first_batch in train_generator:
        model(first_batch[0])
        break

    # Train the model.
    # model.fit(train_generator, epochs=1)


if __name__ == "__main__":
    print("====== Launching train script =======")
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the DLRM-DCNv2 model on the Criteo dataset (MLPerf)"
        )
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the YAML config file."
    )
    args = parser.parse_args()

    print(f"===== Reading config from {args.config_path} ======")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Config: {config}")

    # === Unpack args from config ===

    # == Dataset config ==
    ds_cfg = config["dataset"]
    # File path
    file_pattern = ds_cfg["file_pattern"]
    # Features
    label = ds_cfg["label"]
    dense_features = ds_cfg["dense"]
    emb_features = ds_cfg["sparse"]

    # == Model config ==
    model_cfg = config["model"]
    # Embedding
    embedding_dim = model_cfg["embedding_dim"]
    allow_id_dropping = model_cfg["allow_id_dropping"]
    embedding_threshold = model_cfg["embedding_threshold"]
    max_ids_per_partition = model_cfg["max_ids_per_partition"]
    max_unique_ids_per_partition = model_cfg["max_unique_ids_per_partition"]
    embedding_learning_rate = model_cfg["learning_rate"]
    # MLP
    bottom_mlp_dims = model_cfg["bottom_mlp_dims"]
    top_mlp_dims = model_cfg["top_mlp_dims"]
    # DCN
    num_dcn_layers = model_cfg["num_dcn_layers"]
    dcn_projection_dim = model_cfg["dcn_projection_dim"]

    # == Training config ==
    training_cfg = config["training"]
    learning_rate = training_cfg["learning_rate"]
    global_batch_size = training_cfg["global_batch_size"]
    num_epochs = training_cfg["num_epochs"]

    # For features which have vocabulary_size < embedding_threshold, we can
    # just do a normal dense lookup for those instead of have distributed
    # embeddings. We could ideally pass `placement = default_device` to
    # `keras_rs.layers.TableConfig` directly (and wouldn't have to do this
    # separation of features), but doing it that way will necessarily require
    # a separate optimiser for the embedding layer.
    small_emb_features = []
    large_emb_features = []
    for emb_feature in emb_features:
        if emb_feature["vocabulary_size"] < embedding_threshold:
            small_emb_features.append(emb_feature)
        else:
            large_emb_features.append(emb_feature)

    print(f"{small_emb_features=}")
    print(f"{large_emb_features=}")

    main(
        file_pattern,
        dense_features,
        large_emb_features,
        small_emb_features,
        label,
        embedding_dim,
        allow_id_dropping,
        max_ids_per_partition,
        max_unique_ids_per_partition,
        embedding_learning_rate,
        bottom_mlp_dims,
        top_mlp_dims,
        num_dcn_layers,
        dcn_projection_dim,
        learning_rate,
        global_batch_size,
        num_epochs,
    )
