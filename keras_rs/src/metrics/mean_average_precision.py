import keras

from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.metrics.MeanAveragePrecision")
class MeanAveragePrecision(keras.metrics.Metric):
    def call(self) -> None:
        pass
