from pathlib import Path
from chess import Board
from typing import Optional, Protocol, Union

class ValueFunction(Protocol):
    def __call__(self, board: Board):
        ...
