import random
from copy import deepcopy
from typing import Optional

BOARD_T = list[list[int]]


def get_neighbors_sum(board: BOARD_T, y: int, x: int) -> int:
    up = get_direction(board, "up", y, x)
    down = get_direction(board, "down", y, x)
    left = get_direction(board, "left", y, x)
    right = get_direction(board, "right", y, x)
    top_right_corner = get_direction(board, "right", *up)
    top_left_corner = get_direction(board, "left", *up)
    bottom_right_corner = get_direction(board, "right", *down)
    bottom_left_corner = get_direction(board, "left", *down)
    return sum(
        [
            board[i][j]
            for i, j in [
                up,
                down,
                left,
                right,
                top_right_corner,
                bottom_right_corner,
                top_left_corner,
                bottom_left_corner,
            ]
        ]
    )


def get_direction(board: BOARD_T, direction: str, y: int, x: int) -> tuple[int, int]:
    if direction == "up":
        return y - 1, x
    elif direction == "right":
        return (y, x + 1) if x < len(board[0]) - 1 else (y, 0)
    elif direction == "down":
        return (y + 1, x) if y < len(board) - 1 else (0, x)
    elif direction == "left":
        return y, x - 1
    else:
        raise ValueError("Invalid direction")


def next_generation(board: BOARD_T) -> BOARD_T:
    """
    This function gets a game of life board and return the next generation
    """
    new_board = deepcopy(board)

    for i in range(len(board)):
        for j in range(len(board[0])):
            neighbors_sum = get_neighbors_sum(board, i, j)
            if board[i][j]:
                if neighbors_sum < 2 or neighbors_sum > 3:
                    new_board[i][j] = 0
            else:
                if neighbors_sum == 3:
                    new_board[i][j] = 1

    return new_board


def add_glider(board: BOARD_T, y: int, x: int) -> None:
    board[y][x + 1] = 1
    board[y + 1][x + 2] = 1
    board[y + 2][x] = 1
    board[y + 2][x + 1] = 1
    board[y + 2][x + 2] = 1


def add_cells_pattern(board: BOARD_T, cells_pattern: str, y: int, x: int) -> None:
    data = cells_pattern.split("\n")
    for line in data:
        if line and line[0] in (".", "O"):
            x_temp = x
            for char in line:
                if char == "O":
                    board[y][x_temp] = 1
                x_temp += 1
            y += 1


def generate_random_board(y: int, x: Optional[int] = None) -> BOARD_T:
    if x is None:
        x = y
    return [[random.randint(0, 1) for _ in range(x)] for _ in range(y)]


def generate_empty_board(y: int, x: Optional[int] = None) -> BOARD_T:
    if x is None:
        x = y
    return [[0 for _ in range(x)] for _ in range(y)]
