import yaml
import numpy as np

from training.train import run_training
from evaluation.predict import predict_file_with_timestamps
from preprocessing.load_data import (get_file_paths, load_attack_metadata)

def print_statistics(name, scores):
    print(f"\n{name}")
    print("  Samples:", len(scores))
    print("  Mean:", scores.mean())
    print("  Median:", np.median(scores))
    print("  Min:", scores.min())
    print("  Max:", scores.max())
    print("  P1:", np.percentile(scores, 1))
    print("  P5:", np.percentile(scores, 5))
    print("  P10:", np.percentile(scores, 10))

# ============================================================
# CONFIG
# ============================================================

with open("config/dataset.yaml") as file:
    dataset_config = yaml.safe_load(file)

# CHANGE THIS ARCHIVE WHEN TESTING
with open("config/model_test.yaml") as file:
    model_config = yaml.safe_load(file)


# ============================================================
# TRAINING
# ============================================================

model, history, id_to_index, index_to_id, X_val, y_val = run_training(
    dataset_config,
    model_config
)

# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

print("\n========================================")
print("BENIGN VALIDATION")
print("========================================")

predictions = model.predict(X_val)

validation_scores = predictions[np.arange(len(y_val)), y_val]

print_statistics("BENIGN VALIDATION", validation_scores)

thresholds = {
    "1%": np.percentile(validation_scores, 1),
    "5%": np.percentile(validation_scores, 5),
    "10%": np.percentile(validation_scores, 10),
}

print("\nThresholds:")

for name, threshold in thresholds.items():
    print(f"  {name}: {threshold:.8f}")

# ============================================================
# ATTACK FILES
# ============================================================

attack_files = dataset_config["dataset"]["test"]["files"]

attack_metadata = load_attack_metadata(
    dataset_config["dataset"]["path"]
)

test_file_paths = get_file_paths(
    dataset_config["dataset"]["path"],
    dataset_config["dataset"]["test"]["directory"],
    attack_files
)

# ============================================================
# EVALUATION
# ============================================================

for test_file in test_file_paths:

    print("\n\n========================================")
    print("Arquivo:", test_file.name)
    print("========================================")

    timestamps, y, probabilities = predict_file_with_timestamps(
        model,
        test_file,
        id_to_index,
        model_config["sliding_window"]["n"]
    )

    print("Samples:", len(probabilities))

    # --------------------------------------------------------
    # Threshold evaluation
    # --------------------------------------------------------

    for threshold_name, threshold in thresholds.items():

        anomalies = probabilities < threshold

        anomaly_rate = np.mean(anomalies)

        print(
            f"\nThreshold {threshold_name} "
            f"({threshold:.8f})"
        )

        print("  Anomalies:", anomalies.sum())
        print("  Normal:", (~anomalies).sum())
        print("  Anomaly rate:", anomaly_rate)


    # --------------------------------------------------------
    # Temporal analysis
    # --------------------------------------------------------

    metadata = attack_metadata[test_file.stem]
    injection_interval = metadata["injection_interval"]

    if injection_interval is None:

        print("\nInjection interval: None")
        print("Temporal analysis skipped.")

        continue

    injection_start, injection_end = injection_interval

    print(
        "\nInjection interval:",
        injection_start,
        "->",
        injection_end
    )

    before = probabilities[
        timestamps < injection_start
    ]

    during = probabilities[
        (timestamps >= injection_start) &
        (timestamps <= injection_end)
    ]

    after = probabilities[
        timestamps > injection_end
    ]

    print_statistics("BEFORE", before)
    print_statistics("DURING", during)
    print_statistics("AFTER", after)
