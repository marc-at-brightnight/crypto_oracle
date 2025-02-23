from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras import backend as K  # type: ignore


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from csv file and return dataframe
    """
    return pd.read_csv(file_path)


def reset_random_seeds(seed: int = 1) -> None:
    import os
    import random
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
