import chess
from random import Random
from time import sleep
from mmEngine.value_funtions.value_function import value_function_path
from mmEngine.value_funtions.nn_pytorch import load_model
from mmEngine.value_funtions import MaterialCount, NNPytorchValueFunction
from mmEngine.agents import RandomAgent, MinMaxAgent, MinMaxAlphaBetaAgent, Agent


def main():
    board = chess.Board()

    print("game started... ")
    torch_function = "nn.torch"
    model_path = value_function_path(name=torch_function)
    bot_white = MinMaxAlphaBetaAgent(
        evaluation_function=NNPytorchValueFunction(model=load_model(model_path)),
        depth=3,
    )
    bot_black = RandomAgent()
    for i in range(40):
        if board.is_game_over():
            break

        assert board.turn
        white_move = bot_white.select_move(board)
        if white_move is None:
            break
        print(f"white picked move {board.san(white_move)}")
        board.push(white_move)

        assert not board.turn
        black_move = bot_black.select_move(board)
        print(f"black picked move {board.san(black_move)}")
        board.push(black_move)

    print(board)
    (white, black) = MaterialCount(board)
    print(f"white score is {white}")
    print(f"black score is {black}")

    outcome = board.outcome()
    if outcome is not None:
        print(f"The reason for termination: {outcome.termination}")
        winner = outcome.winner
        if winner is not None:
            print(f"The winner is: {winner}")


if __name__ == "__main__":
    main()