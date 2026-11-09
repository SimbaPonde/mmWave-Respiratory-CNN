"""1D CNN architectures for mmWave breathing-pattern classification."""

from dataclasses import dataclass, asdict

from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class CNNConfig:
    """Hyperparameters for the Conv-Conv-Pool-Dense 1D CNN."""

    name: str = "gap_dropout"
    conv1_filters: int = 32
    conv1_kernel: int = 5
    conv2_filters: int = 64
    conv2_kernel: int = 5
    pool_size: int = 2
    head: str = "gap"  # "gap" (GlobalAveragePooling1D) or "flatten"
    dense_units: int = 64
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 50
    early_stopping_patience: int = 5  # 0 disables early stopping

    def to_dict(self) -> dict:
        return asdict(self)


# The five configurations explored during development. "gap_dropout" is the
# final model: replacing Flatten with GlobalAveragePooling1D and adding dropout
# cut the parameter count by >99% and removed the overfitting seen in the others.
CONFIGS = {
    "baseline": CNNConfig("baseline", 32, 7, 64, 5, 2, "flatten", 128, 0.0, 1e-3, 32, 20, 0),
    "small": CNNConfig("small", 16, 5, 32, 3, 2, "flatten", 64, 0.0, 1e-3, 16, 20, 0),
    "small_kernels": CNNConfig("small_kernels", 32, 3, 64, 3, 2, "flatten", 64, 0.0, 5e-4, 16, 20, 0),
    "short_training": CNNConfig("short_training", 32, 5, 64, 3, 2, "flatten", 128, 0.0, 1e-3, 16, 10, 0),
    "gap_dropout": CNNConfig(),
}


def build_model(cfg: CNNConfig, input_len: int, n_classes: int) -> keras.Model:
    """Build and compile the 1D CNN described by ``cfg``."""
    model = keras.Sequential(name=cfg.name)
    model.add(keras.Input(shape=(input_len, 1)))
    model.add(layers.Conv1D(cfg.conv1_filters, cfg.conv1_kernel, padding="same", activation="relu"))
    model.add(layers.Conv1D(cfg.conv2_filters, cfg.conv2_kernel, padding="same", activation="relu"))
    model.add(layers.MaxPooling1D(cfg.pool_size))
    if cfg.head == "gap":
        model.add(layers.GlobalAveragePooling1D())
    elif cfg.head == "flatten":
        model.add(layers.Flatten())
    else:
        raise ValueError(f"Unknown head: {cfg.head!r}")
    model.add(layers.Dense(cfg.dense_units, activation="relu"))
    if cfg.dropout > 0:
        model.add(layers.Dropout(cfg.dropout))
    model.add(layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
