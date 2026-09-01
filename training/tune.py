import optuna
import tensorflow as tf

from models.gru import GRUModel
from optuna.integration import TFKerasPruningCallback


# Replace these imports with your dataset-loading code
from data import X_train, y_train, X_val, y_val, num_ids


def objective(trial):
    embedding_dim = trial.suggest_int(
        "embedding_dim", 16, 256, step=16
    )

    gru_units = trial.suggest_int(
        "gru_units", 32, 256, step=32
    )

    dropout = trial.suggest_float(
        "dropout", 0.0, 0.5, step=0.1
    )

    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2, log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256]
    )

    model = GRUModel(
        num_ids=num_ids,
        embedding_dim=embedding_dim,
        gru_units=gru_units,
        dropout=dropout,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    callbacks = [
        TFKerasPruningCallback(
            trial,
            monitor="val_loss",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    return min(history.history["val_loss"])


study = optuna.create_study(
    study_name="gru_hyperparameter_optimization",
    storage="sqlite:///gru_optuna.db",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
    ),
)

study.optimize(
    objective,
    n_trials=50,
    timeout=None,
)

print("Best validation loss:", study.best_value)
print("Best hyperparameters:")

for parameter, value in study.best_params.items():
    print(f"{parameter}: {value}")