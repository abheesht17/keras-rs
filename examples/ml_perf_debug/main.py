import argparse
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import yaml
from dataloader import create_dummy_dataset
from model import DLRMDCNV2

import keras_rs

SEED = 1337


def main(
    file_pattern,
    dense_features,
    sparse_features,
    dense_lookup_features,
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
    distribution = keras.distribution.DataParallel()
    keras.distribution.set_distribution(distribution)

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
    print("===== Initialising model =====")
    model = DLRMDCNV2(
        sparse_feature_configs=feature_configs,
        dense_lookup_features=dense_lookup_features,
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
        batch_size=global_batch_size,
        sparse_features=sparse_features,
        dense_lookup_features=dense_lookup_features,
    )
    distribution.distribute_dataset(train_ds)
    distribution.auto_shard_dataset = False

    for first_batch in train_ds:
        print(first_batch)
        break

    def generator(dataset):
        for example in dataset:
            to_yield_x = {
                "dense_features": example["dense_features"],
                "preprocessed_sparse_features": model.embedding_layer.preprocess(
                    example["sparse_features"], training=False
                ),
            }
            if "dense_lookups" in example:
                to_yield_x["dense_lookups"] = example["dense_lookups"]
            to_yield_y = example["clicked"]
            yield to_yield_x, to_yield_y

    train_generator = generator(train_ds)
    for first_batch in train_generator:
        print("--->", first_batch)
        break

    # # === Print shapes on the current host ===
    # print("\n" + "=" * 30, flush=True)
    # print(f"--- Data Shapes on Host {jax.process_index()} ---", flush=True)
    # print(f"Label shape: {label.shape}", flush=True)
    # print(
    #     f"Dense features shape: {features['dense_features'].shape}", flush=True
    # )

    # print("Preprocessed sparse features:", flush=True)
    # preprocessed_sparse = features["preprocessed_sparse_features"]

    # if isinstance(preprocessed_sparse, dict):
    #     for placement, placement_data in preprocessed_sparse.get(
    #         "preprocessed_inputs_per_placement", {}
    #     ).items():
    #         print(f"  Placement '{placement}':", flush=True)
    #         for table_group, coo_matrix in placement_data.get(
    #             "inputs", {}
    #         ).items():
    #             print(f"    Table Group '{table_group[:30]}...':", flush=True)
    #             if hasattr(coo_matrix, "row_ids"):
    #                 print(
    #                     f"      - row_ids shape: {coo_matrix.row_ids.shape}",
    #                     flush=True,
    #                 )
    #             if hasattr(coo_matrix, "col_ids"):
    #                 print(
    #                     f"      - col_ids shape: {coo_matrix.col_ids.shape}",
    #                     flush=True,
    #                 )
    #             if hasattr(coo_matrix, "values"):
    #                 print(
    #                     f"      - values shape: {coo_matrix.values.shape}",
    #                     flush=True,
    #                 )
    # else:
    #     print(
    #         f"  Unexpected structure type: {type(preprocessed_sparse)}",
    #         flush=True,
    #     )
    # print("=" * 30 + "\n", flush=True)

    # for element in train_ds:
    #     print("--->", element[0])
    #     model(element[0])
    #     break

    # # Build the model.
    # for element in train_ds:
    #     model(element[0])
    #     break
    # Train the model.
    # model.fit(train_ds, epochs=1)
    # train_ds = create_dataset(
    #     file_pattern=file_pattern,
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
    sparse_features = ds_cfg["sparse"]

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
    num_steps = training_cfg["num_steps"]
    num_epochs = training_cfg["num_epochs"]
    log_frequency = training_cfg["log_frequency"]

    # For features which have vocabulary_size < embedding_threshold, we can
    # just do a normal dense lookup for those instead of have distributed
    # embeddings.
    print("===== Removing small embedding tables from `sparse_features` =====")
    dense_lookup_features = []
    for sparse_feature in sparse_features:
        if sparse_feature["vocabulary_size"] < embedding_threshold:
            dense_lookup_features.append(sparse_feature)
            sparse_features.remove(sparse_feature)

    print(f"Dense lookup features: {dense_lookup_features}")
    print(f"Sparse features: {sparse_features}")

    main(
        file_pattern,
        dense_features,
        sparse_features,
        dense_lookup_features,
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
