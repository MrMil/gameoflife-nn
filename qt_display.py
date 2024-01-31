# This was mostly written by ChatGPT
import sys
import threading

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt

from gameoflife import BOARD_T
from network import from_tf, to_tf, GOLNetwork

DEFAULT_WINDOW_SIZE = 750


class GameOfLifeDisplayQT(QMainWindow):
    def __init__(self, board: BOARD_T, window_size: int = DEFAULT_WINDOW_SIZE):
        self._window_size = window_size
        self.app = QApplication(sys.argv)
        super().__init__()
        self.board = board

        self.setWindowTitle("Conway's Game Of Life Neural Network")
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        self.board_label = QLabel(self)
        self.layout.addWidget(self.board_label)

        self.update_board(self.board)
        self.board_label.adjustSize()

        self.show()

    def update_board(self, new_board: BOARD_T) -> None:
        self.board = new_board
        height = len(self.board)
        width = len(self.board[0])

        # Create a QPixmap to display the board
        pixmap = QPixmap(width * 2, height * 2)
        painter = QPainter(pixmap)
        painter.fillRect(pixmap.rect(), Qt.white)  # Fill background with white

        painter.setBrush(QColor(Qt.black))
        for row in range(height):
            for col in range(width):
                if self.board[row][col]:
                    painter.drawRect(col * 2, row * 2, 1, 1)

        painter.end()

        # Scale the pixmap to maintain the aspect ratio
        scaled_pixmap = pixmap.scaled(
            self._window_size, self._window_size, Qt.KeepAspectRatio
        )
        self.board_label.setPixmap(scaled_pixmap)


def run_board_with_display(board: BOARD_T, model: GOLNetwork) -> None:
    display = GameOfLifeDisplayQT(board)
    thread = threading.Thread(
        target=iterate_board, args=(display, board, model), daemon=True
    )
    thread.start()
    sys.exit(display.app.exec_())


def iterate_board(
    display: GameOfLifeDisplayQT, board: BOARD_T, model: GOLNetwork
) -> None:
    while True:
        board = from_tf(model.predict(to_tf(board)))
        display.update_board(board)
