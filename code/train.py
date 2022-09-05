from mmEngine.value_funtions import CreateModel, TrainModel, LoadModel
from mmEngine.database import get_database_dir
from pathlib import Path
import numpy as np


def main():
    database_dir = get_database_dir()
    data = []
    for i in range(6):
        processed_database_path = Path(
            database_dir, f"database_processed{i}.npz")
        if processed_database_path.exists:
            data.append(np.load(processed_database_path))

    network_file_path = Path("./trial_network.keras")

    if not network_file_path.exists():
        print(f"{network_file_path} does not exist, creating new model...")
        model = CreateModel()
    else:
        print(f"Loading model at {network_file_path}...")
        model = LoadModel(network_file_path)

    print("Training model...")
    TrainModel(model, data, save_path=network_file_path)


if __name__ == "__main__":
    main()
