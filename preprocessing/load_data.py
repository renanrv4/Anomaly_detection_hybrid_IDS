from pathlib import Path

import pandas as pd
import numpy as np

def get_file_paths(road_path, directory, filenames):
    """
    Get paths for the files specified in the configuration.

    Parameters
    ----------
    road_path : str
        Path to the ROAD dataset.

    filenames : list[str]
        File names without the .csv extension.

    directory : str
        Dataset directory, e.g. "ambient" or "attacks".

    Returns
    -------
    list[Path]
        Paths to the selected CSV files.
    """

    complete_path = Path(road_path) / "signal_extractions" / directory

    return [
        complete_path	 / f"{filename}.csv"
        for filename in filenames
    ]

def load_ids(file_path):
    """
    Load CAN IDs from a ROAD CSV file.
    """

    df = pd.read_csv(file_path)

    return df["ID"].to_numpy()

def create_id_mapping(ids):
    """
    Create mappings between CAN IDs and integer indices.
    """
    unique_ids = sorted(set(ids))

    id_to_index = {
        can_id: index
        for index, can_id in enumerate(unique_ids)
    }

    index_to_id = {
        index: can_id
        for can_id, index in id_to_index.items()
    }

    return id_to_index, index_to_id

def encode_ids(ids, id_to_index):
    """
    Convert CAN IDs to integer indices using the ID mapping.
    """

    return np.array(
        [id_to_index[can_id] for can_id in ids],
        dtype=np.int32
    )

def load_multiple_ids(file_paths):
    """
    Load and concatenate CAN IDs from multiple ROAD CSV files.
    """

    ids = [
        load_ids(file_path)
        for file_path in file_paths
    ]

    return np.concatenate(ids)
