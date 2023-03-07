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

from mmEngine.value_funtions.value_function import ValueFunction, value_function_path


def save_model(
    model: nn.Module, optimizer: optim.Optimizer, loss: float, model_path: Path
):
    torch.save(
        {
            "title": "pytorch chess model",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )


def load_model(model_path: Path) -> nn.Module:
    """
    Load a model from a path.
    
    Args:
        model_path (Path): Path to the model.
    return:
        nn.Module: The loaded model.
    """
    checkpoint = torch.load(model_path)
    model = ChessModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class ChessModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 8*8 board: big kernels -> only few channels
        # start with 13 channels
        # 6 white pieces, 6 black pieces, and no piece = 13
        self.conv1_8_8 = nn.Conv2d(
            in_channels=13, out_channels=13, kernel_size=1, padding="same"
        )
        self.dropout1_8_8 = nn.Dropout2d(0.5)
        self.conv2_8_8 = nn.Conv2d(
            in_channels=13, out_channels=16, kernel_size=3, padding="same"
        )
        self.dropout2_8_8 = nn.Dropout2d(0.5)
        self.conv3_8_8 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding="same"
        )

        # If we want same padding -> https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121
        # Doesn't seem a problem here as the size fits at the moment.
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # 4*4 boards
        self.conv1_4_4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=2, padding="same"
        )
        self.dropout1_4_4 = nn.Dropout2d(0.5)
        self.conv2_4_4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2, padding="same"
        )
        self.dropout2_4_4 = nn.Dropout2d(0.5)
        self.conv3_4_4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=2, padding="same"
        )

        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 2*2 boards
        # this seems serious overkill, I need to plot the response of these things check if
        # they end up producing useful patterns.
        self.conv1_2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, padding="same"
        )
        self.dropout1_2_2 = nn.Dropout2d(0.5)
        self.conv2_2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, padding="same"
        )
        self.dropout2_2_2 = nn.Dropout2d(0.5)
        self.conv3_2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, padding="same"
        )

        # 2*2 image * 128 channels = 4*128=512
        self.output = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1_8_8(x))
        x = self.dropout1_8_8(x)
        x = F.relu(self.conv2_8_8(x))
        x = self.dropout2_8_8(x)
        x = F.relu(self.conv3_8_8(x))

        x = self.max_pool1(x)

        x = F.relu(self.conv1_4_4(x))
        x = self.dropout1_4_4(x)
        x = F.relu(self.conv2_4_4(x))
        x = self.dropout2_4_4(x)
        x = F.relu(self.conv3_4_4(x))

        x = self.max_pool2(x)

        x = F.relu(self.conv1_2_2(x))
        x = self.dropout1_2_2(x)
        x = F.relu(self.conv2_2_2(x))
        x = self.dropout2_2_2(x)
        x = F.relu(self.conv3_2_2(x))

        x = torch.flatten(x, start_dim=1)
        x = self.output(x)

        return x


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
    model_path: Optional[Path] = None,
    disable_save=False,
    disable_load=False,
    disable_wandb=False,
):
    """
    Train a pytorch model on the given dataset.

    args:
        numpy_dataset: a list of numpy arrays containing the dataset.
        model_path: the path to the model to load.
        disable_save: if true, don't save the model.
        disable_load: if true, don't load the model, but always train from scratch.
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

    model: nn.Module = ChessModel().cuda()
    if model_path is not None and not disable_load:
        print(f"Loading existing model at {model_path} \n")
        model = load_model(model_path).cuda()
        print(f"Successfully loaded model at {model_path} \n")

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

        if model_path is not None and not disable_save:
            print(f"-> saving model to {model_path}.")
            save_model(
                model=model,
                model_path=model_path,
                loss=running_loss,
                optimizer=optimizer,
            )

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
