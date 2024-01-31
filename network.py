import tensorflow as tf
import numpy as np

from gameoflife import next_generation, generate_random_board, BOARD_T


class GOLNetwork(tf.keras.Model):
    def __init__(self, filters: int, activation: str) -> None:
        super(GOLNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            activation=activation,
            strides=1,
            kernel_initializer="random_normal",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            1,
            1,
            padding="same",
            strides=1,
            kernel_initializer="random_normal",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        padded = self._add_torus_padding(inputs)
        x = self.conv1(padded)
        x = self.conv2(x)
        return x[:, 1:-1, 1:-1, :]

    @staticmethod
    def _add_torus_padding(x: tf.Tensor) -> tf.Tensor:
        _, height, width, _ = x.shape
        return tf.tile(x, [1, 3, 3, 1])[
            :, (height - 1) : ((height * 2) + 1), (width - 1) : ((width * 2) + 1), :
        ]


def convert_board_to_tf_input(board: BOARD_T) -> np.ndarray:
    return np.array([[[item] for item in row] for row in board], dtype=np.float32)


def to_tf(board: BOARD_T) -> np.ndarray:
    return np.array([convert_board_to_tf_input(board)])


def from_tf(prediction: np.ndarray, threshold: float = 0.2) -> BOARD_T:
    return [[0 if x < threshold else 1 for x in row] for row in prediction[0, :, :, 0]]


def get_trained_model(
    filters: int,
    activation: str,
    dimensions: int,
    number_of_boards: int,
    batch_size: int,
    epochs: int,
) -> GOLNetwork:
    model = GOLNetwork(filters, activation)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    inputs = [generate_random_board(dimensions) for _ in range(number_of_boards)]
    outputs = np.array(
        [
            convert_board_to_tf_input(next_generation(input_board))
            for input_board in inputs
        ]
    )
    inputs = np.array(
        [convert_board_to_tf_input(input_board) for input_board in inputs]
    )
    model.fit(inputs, outputs, epochs=epochs, verbose=1, batch_size=batch_size)
    return model
