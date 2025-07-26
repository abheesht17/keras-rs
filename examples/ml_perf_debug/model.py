import keras
from keras import ops

import keras_rs


def _clone_initializer(initializer: keras.initializers.Initializer, seed):
    config = initializer.get_config()
    config.pop("seed")
    config = {**config, "seed": seed}
    initializer_class: type[keras.initializers.Initializer] = (
        initializer.__class__
    )
    return initializer_class.from_config(config)


class DLRMDCNV2(keras.Model):
    def __init__(
        self,
        sparse_feature_configs,
        embedding_dim,
        bottom_mlp_dims,
        top_mlp_dims,
        num_dcn_layers,
        dcn_projection_dim,
        seed=None,
        dtype=None,
        name=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, name=name, **kwargs)
        self.seed = seed

        # === Layers ====

        # Bottom MLP for encoding dense features
        self.bottom_mlp = keras.Sequential(
            self._get_mlp_layers(
                dims=bottom_mlp_dims,
                intermediate_activation="relu",
                final_activation="relu",
            ),
            name="bottom_mlp",
        )
        # Distributed embeddings for encoding sparse inputs
        self.embedding_layer = keras_rs.layers.DistributedEmbedding(
            feature_configs=sparse_feature_configs,
            dtype=dtype,
            name="embedding_layer",
        )
        # DCN for "interactions"
        self.dcn_block = DCNBlock(
            num_layers=num_dcn_layers,
            projection_dim=dcn_projection_dim,
            seed=seed,
            dtype=dtype,
            name="dcn_block",
        )
        # Top MLP for predictions
        self.top_mlp = keras.Sequential(
            self._get_mlp_layers(
                dims=top_mlp_dims,
                intermediate_activation="relu",
                final_activation="sigmoid",
            ),
            name="top_mlp",
        )

        # === Passed attributes ===
        self.sparse_feature_configs = sparse_feature_configs
        self.embedding_dim = embedding_dim
        self.bottom_mlp_dims = bottom_mlp_dims
        self.top_mlp_dims = top_mlp_dims
        self.num_dcn_layers = num_dcn_layers
        self.dcn_projection_dim = dcn_projection_dim

    def call(self, inputs):
        # Inputs
        dense_features = inputs["dense_features"]
        sparse_features = inputs["preprocessed_sparse_features"]

        # Embed features.
        dense_output = self.bottom_mlp(dense_features)
        sparse_embeddings = self.embedding_layer(sparse_features)

        # Interaction
        x = ops.concatenate(
            [dense_output, *sparse_embeddings.values()],
            axis=-1,
        )
        x = self.dcn_block(x)

        # Predictions
        outputs = self.top_mlp(x)
        return outputs

    def _get_mlp_layers(
        self,
        dims,
        intermediate_activation,
        final_activation,
    ):
        # Layers.
        initializer = keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
            seed=self.seed,
        )

        layers = [
            keras.layers.Dense(
                units=dim,
                activation=intermediate_activation,
                kernel_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                bias_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                dtype=self.dtype,
            )
            for dim in dims[:-1]
        ]
        layers += [
            keras.layers.Dense(
                units=dims[-1],
                activation=final_activation,
                kernel_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                bias_initializer=_clone_initializer(
                    initializer, seed=self.seed
                ),
                dtype=self.dtype,
            )
        ]
        return layers

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sparse_feature_configs": self.sparse_feature_configs,
                "bottom_mlp_dims": self.bottom_mlp_dims,
                "top_mlp_dims": self.top_mlp_dims,
                "num_dcn_layers": self.num_dcn_layers,
                "dcn_projection_dim": self.dcn_projection_dim,
                "seed": self.seed,
            }
        )
        return config


class DCNBlock(keras.layers.Layer):
    def __init__(self, num_layers, projection_dim, seed, dtype, name, **kwargs):
        super().__init__(dtype=dtype, name=name, **kwargs)

        # Layers
        self.layers = [
            keras_rs.layers.FeatureCross(
                projection_dim=projection_dim,
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                bias_initializer="zeros",
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        # Passed attributes
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.seed = seed

    def call(self, x0):
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "projection_dim": self.projection_dim,
                "seed": self.seed,
            }
        )
        return config
