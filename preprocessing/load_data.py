from pathlib import Path

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
        Dataset dir: "train" or "test".

    Returns
    -------
    list[Path]
        Paths to the selected CSV files.
    """

    complete_path = Path(road_path) / "signal_extractions" / directory

    return [
        split_path / f"{filename}.csv"
        for filename in filenames
    ]


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
