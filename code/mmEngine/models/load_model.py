from pathlib import Path
import torch
import torch.nn as nn


def load_model(model_path: Path, model: nn.Module) -> nn.Module:
    """
    Load a model from a path.

    Args:
        model_path (Path): Path to the model, this path should exist.
        model (nn.Module): An instance of the model to load.
    return:
        nn.Module: The loaded model.
    """
    assert model_path.exists(), f"Model path {model_path} does not exist."
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
