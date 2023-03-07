from pathlib import Path
from chess import Board
from typing import Optional, Protocol, Union

class ValueFunction(Protocol):
    def __call__(self, board: Board):
        ...

def value_function_path(name: str) -> Path:
    value_function_path: Path = Path(__file__).parent
    return value_function_path / name