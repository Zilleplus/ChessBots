from mmEngine import database
from mmEngine.database import load_database, get_database_dir
import numpy as np
from pathlib import Path

database_dir = get_database_dir()

for i_batch in [0, 1, 2, 3, 4, 5]:
    print(f"Generating data of batch {i_batch}")

    processed_data = load_database(num_games=1000000, num_skip_games=1000000*i_batch)
    if processed_data is not None:
        (X, Y) = processed_data
        print(f"Saving array of shape {X.shape}")
        np.savez(Path(database_dir, f"database_processed{i_batch}.npz"), X=X, Y=Y)
