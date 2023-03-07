from mmEngine.value_funtions import TrainPytorchModel, value_function_path
from mmEngine.database import get_database_dir
from pathlib import Path
import numpy as np

def get_data() -> list[np.ndarray]:
    database_dir = get_database_dir()
    data = []
    for i in range(6):
        processed_database_path = Path(
            database_dir, f"database_processed{i}.npz")
        if processed_database_path.exists():
            data.append(np.load(processed_database_path))

    return data

def main_pytorch():
    data = get_data()
    network_file_path = value_function_path(name="nn.torch")
    TrainPytorchModel(data, model_path=network_file_path)

if __name__ == "__main__":
    main_pytorch()