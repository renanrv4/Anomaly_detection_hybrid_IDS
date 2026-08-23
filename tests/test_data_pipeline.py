import yaml

from training.train import prepare_training_data

# LOADING AND PREPARING DATA
with open("config/dataset.yaml") as file:
    config = yaml.safe_load(file)

with open("config/model_test.yaml") as file:
    model_config = yaml.safe_load(file)

dataset = config["dataset"]

X, y, id_to_index, index_to_id = prepare_training_data(
    dataset["path"],
    dataset["train"]["directory"],
    dataset["train"]["files"][:3],
    model_config["sliding_window"]["n"]
)

# TESTING SLIDING WINDOW
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nPrimeira janela:")
print("X:", X[0])
print("y:", y[0])
