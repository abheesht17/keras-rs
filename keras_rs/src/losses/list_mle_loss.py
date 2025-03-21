import keras

from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.losses.ListMLELoss")
class ListMLELoss(keras.losses.Loss):
    def call(self) -> None:
        pass
