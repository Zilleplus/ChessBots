import torch.nn as nn
import torch.optim as optim
import torch
from pathlib import Path

from mmEngine.models.store import model_store


def save_model(
    model: nn.Module, optimizer: optim.Optimizer, loss: float, val_loss: float
) -> tuple[Path, bool]:
    """
    Save a model to a path.
    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer to save.
        loss (float): The loss to save.

    returns:
        path, and bool if successful.
    """
    model_name = type(model).__name__
    (model_path, _) = model_store()[model_name]

    if not model_path.parent.exists():
        return model_path, False
    torch.save(
        {
            "title": "pytorch chess model",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "val_loss": val_loss,
        },
        model_path,
    )

    return model_path, True
