import time
from chess import Board
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import wandb
from mmEngine.database import convert
from mmEngine.models import save_model
from mmEngine.value_funtions.value_function import ValueFunction

class DataCache:
    def __init__(self, data: list[np.ndarray]):
        self.data = data
        self.counts = [data["X"].shape[0] for data in self.data]
        self.cumulative_counts = np.cumsum(self.counts)

    def len(self) -> int:
        return sum([data["X"].shape[0] for data in self.data])

    def isInCache(self, index: int) -> bool:
        return index < self.len()

    def absoluteToRelativeIndex(self, index: int) -> tuple[int, int]:
        """
        Convert an absolute index to a relative index.

        args:
            index: the absolute index.
        returns:
            the relative index and the index of the data array.
        """
        assert index < self.len()
        
        in_range = np.where(self.cumulative_counts > index)[0]
        if len(in_range) == 0:
            raise ValueError("Index out of range")
        else:
            data_set_index = in_range[0]
            if data_set_index == 0:
                return index, data_set_index
            else:
                return index - self.cumulative_counts[data_set_index - 1], data_set_index

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the data at the given index.

        args:
            index: the absolute index.
        returns:
            the data at the given index.
        """
        relative_index, data_set_index = self.absoluteToRelativeIndex(index)
        return self.data[data_set_index]["X"][relative_index], self.data[data_set_index]["Y"][relative_index]
    
    def getSlice(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the data at the given indices.

        args:
            indices: the absolute indices.
        returns:
            the data at the given indices.
        """
        X = np.zeros((len(indices), 8, 8), dtype=np.float32)
        y = np.zeros((len(indices), 1), dtype=np.float32)
        index: int
        for (i, index) in enumerate(indices):
            X[i], y[i] = self.get(index)

        return X, y
            
    def getRandomSlice(self, count: int):
        """
        Get a random slice of the data.

        args:
            count: the number of samples to get.
        returns:
            the data at the given indices.
        """
        indices = np.random.randint(0, self.len(), count)
        return self.getSlice(indices)    

def encode_board(board_positions: torch.Tensor) -> torch.Tensor:
    """
    Encode the board positions into a tensor that can be fed into the model.

    args:
        board_positions: a tensor of shape (number_of_samples, 8, 8)
    returns:
        a tensor of shape (number_of_samples, 8, 8, 12)
    """
    assert board_positions.shape[1] == 8
    assert board_positions.shape[2] == 8

    # one_hot encode the pawn type
    board_positions = F.one_hot(input=board_positions.long(), num_classes=13)

    # The conv2d operation in pytorch wants the shape:
    # (sample, channel, width, height)
    board_positions = torch.transpose(board_positions, 3, 2)
    board_positions = torch.transpose(board_positions, 2, 1)

    return board_positions[:, 1:, :, :]


def TrainPytorchModel(
    X_input: np.ndarray,
    y_input: np.ndarray,
    model: nn.Module,
    model_val_loss: float = float("inf"),
    early_stopping: bool = False,
    disable_save=False,
    disable_wandb=False,
):
    """
    Train a pytorch model on the given dataset.

    args:
        numpy_dataset: a list of numpy arrays containing the dataset.
        model: the model to train.
        model_val_loss: the validation loss of the model.
        early_stopping: if true, stop training if the validation loss doesn't
            improve.
        disable_save: if true, don't save the model.
        disable_wandb: if true, don't log to wandb.

    returns:
        The trained model.
    """
    # assert len(numpy_dataset) > 0

    # number_of_samples: int = sum([data["X"].shape[0] for data in numpy_dataset])
    # print(f"Number of samples: {number_of_samples}")

    batch_size: int = 20000

    X = torch.from_numpy(X_input).float()
    Y = torch.from_numpy(y_input).float().cuda()  # y

    # remove numpy_dataset from memory
    #del numpy_dataset

    # reshape to (sample, width of board, height of board)
    X = torch.reshape(X, (X.shape[0], 8, 8))

    dataset = TensorDataset(X, Y)
    train_set, val_set, test_set = random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

    if not disable_wandb:
        wandb.init(project="chess_pytorch")
        wandb.watch(model, log_freq=100)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = nn.MSELoss()

    num_epochs = 10
    for epoch in range(0, num_epochs):
        running_loss: float = 0.0
        model.train()
        start = time.time()
        for i, sample in enumerate(DataLoader(train_set, batch_size=batch_size)):
            board_positions, win_rate = sample
            board_positions = board_positions.cuda()
            win_rate = win_rate.cuda()
            board_positions = encode_board(board_positions).float()

            optimizer.zero_grad()

            outputs = torch.squeeze(model(board_positions))
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
                X = X.cuda()
                y = y.cuda()

                pred = model(encode_board(X).float())
                pred = torch.squeeze(pred)
                loss = loss_function(pred, y)
                val_loss += loss.item()

        print(f"[{epoch}] loss: {running_loss} val_loss: {val_loss}")
        if not disable_wandb:
            wandb.log({"loss": running_loss})
            wandb.log({"val_loss": val_loss})

        if disable_save:
            print("-> not saving model because disable_save is true.")
        else:
            if val_loss < model_val_loss or not early_stopping:
                model_path, save_success = save_model(
                    model=model,
                    loss=running_loss,
                    optimizer=optimizer,
                    val_loss=val_loss,
                )
                model_val_loss = val_loss
                if save_success:
                    print(f"-> saved model to {model_path}.")
                else:
                    print(f"-> failed to save model to {model_path}.")
            else:
                print(
                    f"-> not saving model because val_loss({val_loss}) is not lower the the previous run ({model_val_loss})."
                )
                print(f"-> stopping training.")
                break

    model.eval()
    test_loss: float = 0.0
    for i, sample in enumerate(DataLoader(test_set, batch_size=batch_size)):
        board_positions, win_rate = sample
        board_positions = board_positions.cuda()
        win_rate = win_rate.cuda()
        board_positions = encode_board(board_positions).float()

        outputs = model(board_positions)
        pred = torch.squeeze(outputs)
        loss = loss_function(pred, y)

        test_loss += loss

    print(f"test loss: {test_loss}")

    return model


class NNPytorchValueFunction(ValueFunction):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        model.training = False

    def __call__(self, board: Board) -> float:
        np_board: np.ndarray = convert(board)
        X = torch.from_numpy(np_board).float()  # shape=(64,)
        X = torch.reshape(X, (1, 8, 8))  # shape=(8, 8)
        X = encode_board(X)  # shape=(12, 8, 8)
        pred = self.model(X)

        if board.turn:
            return float(pred)
        else:
            return -float(pred)
