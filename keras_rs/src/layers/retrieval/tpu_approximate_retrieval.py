import keras

from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.TPUApproximateRetrieval")
class TPUApproximateRetrieval(keras.layers.Layer):
    def call(self) -> None:
        pass
