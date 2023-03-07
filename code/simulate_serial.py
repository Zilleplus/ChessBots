from typing import Optional
import chess
from mmEngine.agents import Agent, RandomAgent, MinMaxAgent
from mmEngine.agents.minMaxAlphaBetaSearch import MinMaxAlphaBetaAgent
from mmEngine.value_funtions import MaterialCount, value_function_path
from mmEngine.value_funtions.nn_pytorch import NNPytorchValueFunction, load_model
from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class Result:
    score_white: int
    score_black: int
    error: Optional[str]
    winner: Optional[bool]

def simulate(agent1: Agent, agent2: Agent, seed: int, num_moves: int = 50):
    np.random.seed(seed)
    b = chess.Board()
    error: Optional[str] = None
    for i in range(num_moves):
        if b.is_game_over():
            break
        m1 = agent1.select_move(b)
        if m1 is None:
            error = "agent1 can't find move"
            break
        else:
            print(f"agent1 selected move {m1}")
        b.push(m1)

        if b.is_game_over():
            break
        m2 = agent2.select_move(b)
        if m2 is None:
            error = "agent2 can't find move"
            break
        else:
            print(f"agent2 selected move {m2}")
        b.push(m2)


    (white_score, black_score) = MaterialCount(b)

    winner: Optional[bool] = None
    outcome = b.outcome()
    if outcome is not None:
        winner = outcome.winner
    return Result(white_score, black_score, error, winner)

def simulate_games(agent1_factory, agent2_factory, num_games: int=10) -> list[Result]:
    results: list[Result] = []
    for i in range(num_games):
        print(f"Simulating game {i}")
        res: Result = simulate(agent1_factory(), agent2_factory(), seed=i)
        print(res)

    avg_white_score = sum([x.score_white for x in results])/len(results)
    print(f"average white score={avg_white_score}")
    avg_black_score = sum([x.score_black for x in results])/len(results)
    print(f"average black score={avg_black_score}")

    white_wins: int = sum([1 if x.winner is not None and x.winner else 0 for x in results])
    print(f"white has {white_wins} wins")

    black_wins: int = sum([1 if x.winner is not None and not x.winner else 0 for x in results])
    print(f"black has {black_wins} wins")

    return results


def main() -> None:
    torch_function = "nn.torch"
    model_path = value_function_path(name=torch_function)
    fac1 = lambda: MinMaxAlphaBetaAgent(evaluation_function=NNPytorchValueFunction(model=load_model(model_path)), depth=3)
    fac2 = lambda: RandomAgent()
    results = simulate_games(fac1, fac2,num_games=4)

if __name__ == "__main__":
    main()