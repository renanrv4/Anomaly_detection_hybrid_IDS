import yaml

from training.train import (
    prepare_training_data,
    create_model,
    compile_model,
    train_model
)

# PREPARING DATA BEFORE TRAINING
with open("config/dataset.yaml") as file:
    dataset_config = yaml.safe_load(file)

with open("config/model_test.yaml") as file:
    model_config = yaml.safe_load(file)

dataset = dataset_config["dataset"]

X, y, id_to_index, index_to_id = prepare_training_data(
    dataset["path"],
    dataset["train"]["directory"],
    dataset["train"]["files"][:3],
    model_config["sliding_window"]["n"]
)

num_ids = len(id_to_index)

model = create_model(
    model_config,
    num_ids
)

# USING JUST A FEW DATA SAMPLES TO TEST MODEL
X_test = X[:1024]
y_test = y[:1024]

num_ids = len(id_to_index)

model = create_model(
    model_config,
    num_ids
)

compile_model(
    model,
    model_config
)

history = train_model(
    model,
    X_test,
    y_test,
    model_config
)
