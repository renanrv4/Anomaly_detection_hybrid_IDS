import tensorflow as tf

from preprocessing.load_data import (get_file_paths, load_multiple_ids, create_id_mapping, encode_ids)

from preprocessing.sliding_window import create_sliding_windows

from models.gru import GRUModel

def prepare_training_data(road_path, directory, filenames, n):
    """
    Load and encode the training CAN ID sequence.
    """

    file_paths = get_file_paths(
        road_path,
        directory,
        filenames
    )

    ids = load_multiple_ids(file_paths)

    id_to_index, index_to_id = create_id_mapping(ids)

    encoded_ids = encode_ids(
        ids,
        id_to_index
    )

    X, y = create_sliding_windows(
        encoded_ids,
        n
    )

    return X, y, id_to_index, index_to_id

def create_model(config, num_ids):
    """
    Create the GRU model from the configuration.
    """

    model = GRUModel(
        num_ids=num_ids,
        embedding_dim=config["model"]["embedding_dim"],
        gru_units=config["model"]["gru_units"],
        dropout=config["model"]["dropout"]
    )

    return model

def compile_model(model, config):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

def train_model(model, X, y, config):
    history = model.fit(
        X,
        y,
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"]
    )

    return history

def run_training(dataset_config, model_config):
    """
    Execute the complete training pipeline.
    """

    dataset = dataset_config["dataset"]

    # 1. Prepare training data
    X, y, id_to_index, index_to_id = prepare_training_data(
        dataset["path"],
        dataset["train"]["directory"],
        dataset["train"]["files"],
        model_config["sliding_window"]["n"]
    )

    # 2. Create model
    num_ids = len(id_to_index)

    model = create_model(
        model_config,
        num_ids
    )

    # 3. Compile model
    compile_model(
        model,
        model_config
    )

    # 4. Train model
    history = train_model(
        model,
        X,
        y,
        model_config
    )

    return model, history, id_to_index, index_to_id
