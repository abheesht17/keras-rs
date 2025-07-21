import argparse
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import yaml
from dataloader import create_dummy_dataset
from model import DLRMDCNV2

import keras_rs


def main(
    file_pattern,
    dense_features,
    sparse_features,
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
    num_steps,
    num_epochs,
    log_frequency,
):
    # Set DDP as Keras distribution strategy
    data_parallel = keras.distribution.DataParallel()
    keras.distribution.set_distribution(data_parallel)

    # === Distributed embeddings' configs for sparse features ===
    feature_configs = {}
    for sparse_feature in sparse_features:
        # TODO: We don't need this custom renaming. Remove later, when we
        # shift from dummy data to actual data.
        feature_name = (
            sparse_feature["name"]
            .replace("-", "_")
            .replace("egorical_feature", "")
        )
        vocabulary_size = sparse_feature["vocabulary_size"]

        table_config = keras_rs.layers.TableConfig(
            name=f"{feature_name}_table",
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            # TODO(abheesht): Verify.
            initializer=keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_in", distribution="uniform"
            ),
            optimizer=keras.optimizers.Adagrad(
                learning_rate=embedding_learning_rate
            ),
            combiner="sum",
            placement="sparsecore",
            max_ids_per_partition=max_ids_per_partition,
            max_unique_ids_per_partition=max_unique_ids_per_partition,
        )
        feature_configs[f"{feature_name}_id"] = keras_rs.layers.FeatureConfig(
            name=feature_name.replace("id", ""),
            table=table_config,
            # TODO(abheesht): Verify whether it should be `(bsz, 1)` or
            # `(bsz, multi_hot_size)`.
            input_shape=(global_batch_size, 1),
            output_shape=(global_batch_size, embedding_dim),
        )

    # === Instantiate model ===
    # We instantiate the model first, because we need to preprocess sparse
    # inputs using the distributed embedding layer defined inside the model
    # class.
    model = DLRMDCNV2(
        sparse_feature_configs=feature_configs,
        bottom_mlp_dims=bottom_mlp_dims,
        top_mlp_dims=top_mlp_dims,
        num_dcn_layers=num_dcn_layers,
        dcn_projection_dim=dcn_projection_dim,
        dtype="float32",
        name="dlrm_dcn_v2",
    )
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adagrad(learning_rate=learning_rate),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    # === Load dataset ===
    train_ds = create_dummy_dataset(
        batch_size=global_batch_size,
        multi_hot_sizes=[
            feature["multi_hot_size"] for feature in sparse_features
        ],
        vocabulary_sizes=[
            feature["vocabulary_size"] for feature in sparse_features
        ],
        sparse_feature_preprocessor=model.embedding_layer,
    )()
    model.fit(train_ds, epochs=1)
    # train_ds = create_dataset(
    #     file_pattern=file_pattern,
    #     sparse_feature_preprocessor=model.embedding_layer,
    #     per_replica_global_batch_size=global_batch_size,
    #     dense_features=dense_features,
    #     sparse_features=[f["name"] for f in sparse_features],
    #     multi_hot_sizes=[f["multi_hot_size"] for f in sparse_features],
    #     vocabulary_sizes=[f["vocabulary_size"] for f in sparse_features],
    #     label=label,
    #     num_processes=1,
    #     process_id=0,
    #     parallelism=1,
    #     training=True,
    #     return_dummy_dataset=True,
    # )()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the DLRM-DCNv2 model on the Criteo dataset (MLPerf)"
        )
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the YAML config file."
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # === Unpack args from config ===

    # == Dataset config ==
    ds_cfg = config["dataset"]
    # File path
    file_pattern = ds_cfg["file_pattern"]
    # Features
    label = ds_cfg["label"]
    dense_features = ds_cfg["dense"]
    sparse_features = ds_cfg["sparse"]

    # == Model config ==
    model_cfg = config["model"]
    # Embedding
    embedding_dim = model_cfg["embedding_dim"]
    allow_id_dropping = model_cfg["allow_id_dropping"]
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
    num_steps = training_cfg["num_steps"]
    num_epochs = training_cfg["num_epochs"]
    log_frequency = training_cfg["log_frequency"]

    main(
        file_pattern,
        dense_features,
        sparse_features,
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
        num_steps,
        num_epochs,
        log_frequency,
    )
