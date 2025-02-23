import os
from pathlib import Path
from typing import Callable

from keras import Sequential

from src.models.cnn_lstm_model import build_cnn_lstm
from src.models.predict import predict
from src.models.train import train_model
from src.data.data_cleaning import preprocess_data
from src.data.data_preparation import (
    prepare_data,
    save_preprocessed_data,
)
from src.data.feature_engineering import (
    create_binary_classification,
    select_features,
    save_selected_features,
)
from src.io_models.inputs import Inputs, Model
from src.io_models.outputs import Outputs
from src.models.lstnet_model import build_lstnet
from src.models.tcn_model import build_tcn
from src.utils.project_functions import load_data, reset_random_seeds


MODEL_MAP: dict[Model, Callable[[tuple[int, int]], Sequential]] = {
    "cnn_lstm": build_cnn_lstm,
    "lstnet": build_lstnet,
    "tcn": build_tcn,
}


def main(inputs: Inputs) -> Outputs:
    # Data cleaning and preprocessing
    print(f"Loading data from {inputs.raw_data_path}...")
    data = load_data(inputs.raw_data_path)
    print("Data loaded. Starting preprocessing...")
    cleaned_data = preprocess_data(data)
    os.makedirs(os.path.dirname(inputs.cleaned_data_path), exist_ok=True)
    cleaned_data.to_csv(inputs.cleaned_data_path, index=False)
    print(f"Cleaned data saved to {inputs.cleaned_data_path}")

    # Feature engineering
    df = load_data(inputs.cleaned_data_path)
    y = create_binary_classification(df)
    features_selected, features_selected_tentative = select_features(df, y)
    print("Features selected:", features_selected)
    save_selected_features(df, features_selected, inputs.boruta_data_path)

    # Data preparation
    df = load_data(inputs.boruta_data_path)

    X_train, X_test, y_train, y_test, time_test, price, scaler = prepare_data(
        df, inputs.timesteps
    )
    save_preprocessed_data(
        inputs.trained_data_path,
        X_train,
        X_test,
        y_train,
        y_test,
        time_test,
        price,
        scaler,
    )

    # Model training
    input_shape = (inputs.timesteps, X_train.shape[2])

    reset_random_seeds()
    build_model = MODEL_MAP[inputs.model]
    built_model = build_model(input_shape)
    train_model(
        built_model,
        X_train,
        y_train,
        inputs.epochs,
        inputs.batch_size,
        inputs.model_save_path,
    )

    # Prediction
    return predict(inputs.model_save_path, inputs.trained_data_path)


if __name__ == "__main__":
    outputs = main(Inputs(model="cnn_lstm"))
