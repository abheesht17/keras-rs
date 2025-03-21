import keras

from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.FrequencyEstimator")
class FrequencyEstimator(keras.layers.Layer):
    def call(self) -> None:
        pass
