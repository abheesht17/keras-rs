import keras

from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.metrics.DCG")
class DCG(keras.metrics.Metric):
    def call(self) -> None:
        pass
