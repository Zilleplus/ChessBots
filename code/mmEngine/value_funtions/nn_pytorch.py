from pathlib import Path
import time
from typing import Optional
from chess import Board
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import wandb
from mmEngine.database import convert
from mmEngine.models import load_model, save_model
from mmEngine.models.big_cnn import BigCNN
from mmEngine.models.store import model_store
from mmEngine.value_funtions.value_function import ValueFunction

def encode_board(board_positions: torch.Tensor) -> torch.Tensor:
    """
    Encode the board positions into a tensor that can be fed into the model.
    
    args:
        board_positions: a tensor of shape (number_of_samples, 8, 8)
    returns:
        a tensor of shape (number_of_samples, 8, 8, 13)
    """
    assert board_positions.shape[1] == 8
    assert board_positions.shape[2] == 8

    # one_hot encode the pawn type
    board_positions = F.one_hot(input=board_positions.long(), num_classes=13)

    # The conv2d operation in pytorch wants the shape:
    # (sample, channel, width, height)
    board_positions = torch.transpose(board_positions, 3, 2)
    board_positions = torch.transpose(board_positions, 2, 1)
    board_positions = board_positions.float()

    return board_positions


def TrainPytorchModel(
    numpy_dataset: list[np.ndarray],
    model: nn.Module,
    disable_save=False,
    disable_wandb=False,
):
    """
    Train a pytorch model on the given dataset.

    args:
        numpy_dataset: a list of numpy arrays containing the dataset.
        model: the model to train.
        disable_save: if true, don't save the model.
        disable_wandb: if true, don't log to wandb.

    returns:
        The trained model.
    """
    assert len(numpy_dataset) > 0
    batch_size: int = 20000

    X = torch.from_numpy(numpy_dataset[0]["X"]).float()
    Y = torch.from_numpy(numpy_dataset[0]["Y"]).float().cuda()  # y

    # remove numpy_dataset from memory
    del numpy_dataset

    # reshape to (sample, width of board, height of board)
    X = torch.reshape(X, (X.shape[0], 8, 8))

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

    if not disable_wandb:
        wandb.init(project="chess_pytorch")
        wandb.watch(model, log_freq=100)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = nn.MSELoss()

    num_epochs = 10
    for epoch in range(0, num_epochs):
        running_loss: float = 0.0
        model.train()
        start = time.time()
        for i, sample in enumerate(DataLoader(train_set, batch_size=batch_size)):
            board_positions, win_rate = sample
            board_positions = encode_board(board_positions.cuda())
            win_rate = torch.reshape(win_rate.cuda() ,(win_rate.shape[0], 1))

            optimizer.zero_grad()

            outputs = model(board_positions)
            loss = loss_function(outputs, win_rate)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print the total number of batches in trainset
            if (i + 1) % 100 == 0:
                total_number_of_batches = float(len(train_set) / batch_size)
                # inplace print of the progress
                total_time = time.time() - start
                average_time_per_batch = total_time / (i + 1)
                estimated_time_to_finish = average_time_per_batch * (
                    total_number_of_batches - (i + 1)
                )
                print("\033[2K", end="\r")  # clear the entire line
                print(
                    f"batches used={float(i+1)/ total_number_of_batches * 100:.2f}% estimated time to finish: {int(estimated_time_to_finish/60)} minutes total running time = {int(total_time/60)} minutes",
                    end="\r",
                )
        scheduler.step()

        model.eval()
        val_loss: float = 0.0
        with torch.no_grad():
            for i, sample in enumerate(DataLoader(val_set, batch_size)):
                X, y = sample
                Y = torch.reshape(Y ,(Y.shape[0], 1))
                pred = model(encode_board(X))
                loss = loss_function(pred, y)
                val_loss += loss.item()

        print(f"[{epoch}] loss: {running_loss} val_loss: {val_loss}")
        if not disable_wandb:
            wandb.log({"loss": running_loss})
            wandb.log({"val_loss": val_loss})

        
        model_path, save_sucess = save_model(
            model=model,
            loss=running_loss,
            optimizer=optimizer,
        )
        if save_sucess:
            print(f"-> saved model to {model_path}.")
        else:
            print(f"-> failed to save model to {model_path}.")

    model.eval()
    test_loss: float = 0.0
    for i, sample in enumerate(DataLoader(test_set, batch_size=batch_size)):
        board_positions, win_rate = sample
        board_positions = encode_board(board_positions)

        outputs = model(board_positions)
        loss = loss_function(outputs, win_rate)

        test_loss += loss

    print(f"test loss: {test_loss}")

    return model


class NNPytorchValueFunction(ValueFunction):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        model.training = False

    def __call__(self, board: Board) -> float:
        np_board: np.ndarray = convert(board)
        X = torch.from_numpy(np_board).float() # shape=(64,)
        X = torch.reshape(X, (1, 8, 8)) # shape=(8, 8)
        X = encode_board(X) # shape=(13, 8, 8)
        pred = self.model(X)

        if board.turn:
            return float(pred)
        else:
            return -float(pred)
