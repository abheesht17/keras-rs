import keras
from keras import ops

import keras_rs


def _clone_initializer(initializer: keras.initializers.Initializer):
    config = initializer.get_config()
    initializer_class: type[keras.initializers.Initializer] = (
        initializer.__class__
    )
    return initializer_class.from_config(config)


class DLRMDCNV2(keras.Model):
    def __init__(
        self,
        sparse_feature_configs,
        bottom_mlp_dims,
        top_mlp_dims,
        num_dcn_layers,
        dcn_projection_dim,
        dtype,
        name=None,
        **kwargs,
    ):
        # === Layers ====

        # Bottom MLP for encoding dense features
        self.bottom_mlp = keras.Sequential(
            self._get_mlp_layers(
                dims=bottom_mlp_dims,
                intermediate_activation="relu",
                final_activation="relu",
            ),
            dtype=dtype,
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
            dtype=dtype,
            name="top_mlp",
        )

        # Passed attributes
        self.sparse_feature_configs = sparse_feature_configs
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
        x = ops.concatenate([dense_output, *sparse_embeddings], axis=-1)
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
            scale=1.0, mode="fan_in", distribution="uniform"
        )

        layers = [
            keras.layers.Dense(
                units=dim,
                activation=intermediate_activation,
                kernel_initializer=_clone_initializer(initializer),
                bias_initializer=_clone_initializer(initializer),
            )
            for dim in dims[:-1]
        ]
        layers += [
            keras.layers.Dense(
                units=dims[-1],
                activation=final_activation,
                kernel_initializer=_clone_initializer(initializer),
                bias_initializer=_clone_initializer(initializer),
            )
        ]
        return layers


class DCNBlock(keras.layers.Layer):
    def __init__(self, num_layers, projection_dim, dtype, name, **kwargs):
        super().__init__(dtype=dtype, name=name, **kwargs)

        # Layers
        self.layers = [
            keras_rs.layers.FeatureCross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_normal",
                bias_initializer="zeros",
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        # Passed attributes
        self.num_layers = num_layers
        self.projection_dim = projection_dim

    def call(self, x0):
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        return xl

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "projection_dim": self.projection_dim,
            }
        )
        return config
