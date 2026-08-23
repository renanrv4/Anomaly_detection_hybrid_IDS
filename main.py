import yaml

from training.train import run_training


with open("config/dataset.yaml") as file:
    dataset_config = yaml.safe_load(file)


# CHANGE THIS ARCHIVE WHEN TESTING
with open("config/model_test.yaml") as file:
    model_config = yaml.safe_load(file)


model, history, id_to_index, index_to_id = run_training(
    dataset_config,
    model_config
)
