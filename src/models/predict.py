import os
from pathlib import Path

from src.io_models.outputs import Outputs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.optimizers.legacy import Adam  # type: ignore
from src.utils.project_functions import f1_score
from src.data_processing.data_preparation import load_preprocessed_data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as skl_f1_score,
    roc_auc_score,
    matthews_corrcoef,
)


def predict(model_path: Path, data_path: Path) -> Outputs:
    # Load the preprocessed data
    model_train_data = load_preprocessed_data(data_path)

    # Load the model without compiling it
    model = load_model(model_path, compile=False)

    # Compile the model with custom objects
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall", f1_score],
    )

    # Make predictions
    y_pred_prob = model.predict(model_train_data.x_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Ensure y_test and y_pred are binary arrays
    y_test = model_train_data.y_test.astype(int)
    y_pred = y_pred.astype(int)

    return Outputs(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1_score=skl_f1_score(y_test, y_pred),
        auc_roc_score=roc_auc_score(y_test, y_pred),
        mcc=matthews_corrcoef(y_test, y_pred),
    )
