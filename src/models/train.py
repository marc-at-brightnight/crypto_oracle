import os
from pathlib import Path

import numpy as np
import pandas as pd
from keras import Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.callbacks import EarlyStopping  # type: ignore


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    model_save_path: Path,
) -> None:
    early_stopping = EarlyStopping(monitor="val_loss", patience=100)
    model.fit(
        X_train,
        y_train,
        batch_size,
        epochs,
        validation_split=0.1,
        callbacks=[early_stopping],
    )

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save the model architecture and weights
    model.save(model_save_path)
