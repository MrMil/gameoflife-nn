import argparse

import requests

from gameoflife import generate_empty_board, add_cells_pattern
from network import get_trained_model, from_tf, to_tf, GOLNetwork
from pygame_display import GameOfLifeDisplay

DIMENSIONS = 20


def display_glider_gun(model: GOLNetwork) -> None:
    board = generate_empty_board(120, 150)
    cells_pattern = requests.get(
        "https://conwaylife.com/patterns/gosperglidergun.cells"
    ).text
    add_cells_pattern(board, cells_pattern, 10, 10)
    display = GameOfLifeDisplay(board, 1000)
    while True:
        board = from_tf(model.predict(to_tf(board), verbose=0))
        display.update_board(board)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", action="store_true", default=False, help="Train the model"
    )
    parser.add_argument(
        "-w",
        "--model-weights",
        type=str,
        default="GOLNetwork.ckpt",
        help="Location of the pre-trained model. The default value loads the weights in the repo: GOLNetwork.ckpt",
    )
    parser.add_argument(
        "-s",
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model to a file",
    )
    args = parser.parse_args()

    if args.train:
        model = get_trained_model()
    else:
        model = GOLNetwork()
        model.load_weights(args.model_weights)

    if args.save_model:
        model.save_weights(args.model_weights)

    display_glider_gun(model)


if __name__ == "__main__":
    main()
