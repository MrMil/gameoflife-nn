import matplotlib.pyplot as plt
import numpy as np

from gameoflife import BOARD_T


def draw_board(board: BOARD_T) -> None:
    matrix = np.array(board)
    aspect_ratio = matrix.shape[1] / matrix.shape[0]
    plt.figure(figsize=(5 * aspect_ratio, 5))
    plt.imshow(matrix, cmap="gray_r", interpolation="none")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
