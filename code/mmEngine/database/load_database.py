from pathlib import Path
from os.path import exists
from typing import Any, Dict, Optional, Tuple
from chess.pgn import read_game, read_headers
import numpy as np
import chess

def convert(board: chess.Board) -> np.ndarray:
    # we have 8*8=64 squares(so 64 times a byte)
    x: np.ndarray = np.zeros(8*8, np.uint8)

    for i in range(8*8):
        p: Optional[chess.Piece] = board.piece_at(i)
        if p is None:
            continue
        x[i] = p.piece_type

    # Not sure if this is a smart idea.
    # x[64] = 1 if board.turn else 0

    return x

def get_database_dir() -> Path:
    return Path(__file__).parents[0]

def load_database(num_games: int, num_skip_games: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    database_dir = get_database_dir()
    database_location =   Path(database_dir , "database.pgn")

    if not exists(database_location):
        print(f"cant find database at {database_location}")
        return None

    print(f"Reading database from: {database_location}")

    with open(database_location) as data:
        X: list[np.ndarray] = []
        Y: list[np.int8] = []
        for i_game in range(num_skip_games):
            if i_game%100000 == 0:
                print(f"skipped till game i={i_game}")
            try:
                g = read_game(data)
            except ValueError as e:
                # Just ignore the errors, we can't recover them anyway.
                continue

        for i_game in range(num_games):
            try:
                g = read_game(data)
                if g is None:
                    print("No more games left")
                    break # end of the file
                output_value: Dict[str, int] = {'1/2-1/2':0, '0-1':-1, '1-0':1}

                result: str = g.headers["Result"]
                if result not in output_value:
                    print(f"unkown result on game i={i_game} with headers[\"Results\"]={result}")
                    continue

                y = np.int8(output_value[result])

                board = chess.Board()
                for i, move in enumerate(g.mainline_moves()):
                    board.push(move)
                    x = convert(board)
                    X.append(x)
                    Y.append(y)
            except:
                print(f"error on game i={i_game}")
        
        return (np.array(X), np.array(Y))
