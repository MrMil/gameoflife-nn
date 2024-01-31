# This was mostly written by ChatGPT
import os
import sys

from gameoflife import BOARD_T

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402


class GameOfLifeDisplay:
    def __init__(self, board: BOARD_T, width: int = 800):
        pygame.init()
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0]) if self.rows > 0 else 0
        self.cell_size = width // self.cols
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Conway's Game of Life Neural Network")
        self.update_board(board)

    def draw_grid(self) -> None:
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width, y))

    def update_board(self, new_board: BOARD_T) -> None:
        self._process_quitting()

        self.board = new_board
        self.screen.fill((255, 255, 255))
        self.draw_grid()

        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == 1:
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 0),
                        (
                            col * self.cell_size,
                            row * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )

        pygame.display.flip()

    @staticmethod
    def _process_quitting() -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
