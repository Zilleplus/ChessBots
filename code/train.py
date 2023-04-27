from pathlib import Path

import numpy as np

from mmEngine.database import get_database_dir
from mmEngine.models import load_model
from mmEngine.models.store import model_store
from mmEngine.value_funtions import TrainPytorchModel


def get_data() -> list[np.ndarray]:
    database_dir = get_database_dir()
    data = []
    for i in range(6):
        processed_database_path = Path(database_dir, f"database_processed{i}.npz")
        if processed_database_path.exists():
            data.append(np.load(processed_database_path))

    return data


def main_pytorch():
    data = get_data()
    # model_path, model = model_store()["BigCNN"]
    model_path, model = model_store()["SmallCNN"]
    if model_path.exists():
        print(f"Loading existing model at {model_path} \n")
        model = load_model(model_path, model)
    else:
        print(f"Created new model, no model found at {model_path} \n")

    TrainPytorchModel(data, model=model.cuda())


if __name__ == "__main__":
    main_pytorch()
