import numpy as np

from preprocessing.load_data import (load_multiple_ids, create_id_mapping, encode_ids, load_ids_with_timestamps)
from preprocessing.sliding_window import create_sliding_windows

def predict_file(model, file_path, id_to_index, n):
    """
    Load and encode the attack file CAN ID sequence
    """
    
    list_ids = load_multiple_ids([file_path])
    
    encoded_ids = encode_ids(list_ids, id_to_index)

    X, y = create_sliding_windows(
        encoded_ids,
        n
    )

    predictions = model.predict(X)
    
    # Probability assigned to the actual target ID
    target_probabilities = predictions[
        np.arange(len(y)),
        y
    ]
    return y, target_probabilities

def predict_file_with_timestamps(model, file_path, id_to_index, n):
    """
    Predict target-ID probabilities while preserving center timestamps.
    """

    ids, timestamps = load_ids_with_timestamps(file_path)

    encoded_ids = encode_ids(ids, id_to_index)

    X, y = create_sliding_windows(encoded_ids, n)

    predictions = model.predict(X)

    target_probabilities = predictions[
        np.arange(len(y)),
        y
    ]

    center = n // 2
    center_timestamps = timestamps[center:len(timestamps) - center]

    return center_timestamps, y, target_probabilities

def predict_files(model, file_paths, id_to_index, n):
    results = {}

    for file_path in file_paths:
        results[file_path] = predict_file(
            model,
            file_path,
            id_to_index,
            n
        )

    return results


