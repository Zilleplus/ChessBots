from pathlib import Path
from .big_cnn import BigCNN
from .small_cnn import SmallCNN
from typing import Tuple, cast
from torch import nn


def torch_file_location(name: str) -> Path:
    value_function_path: Path = Path(__file__).parent
    return value_function_path / name


def model_store() -> dict[str, Tuple[Path, nn.Module]]:
    """
    Models that are available to be loaded.

    returns:
        A dictionary of models, the key is the name of the model,
        the value is a tuple of the path to the model and an instance of the model.
    """
    models = {
        SmallCNN.__name__: (
            torch_file_location(name=SmallCNN.__name__ + ".torch"),
            cast(nn.Module, SmallCNN()),
        ),
        BigCNN.__name__: (
            torch_file_location(name=BigCNN.__name__ + ".torch"),
            cast(nn.Module, BigCNN()),
        ),
    }

    return models
