import os
from typing import Callable

from keras import Sequential
from loguru import logger

from src.data_processing.data_cleaning import preprocess_data
from src.data_processing.data_preparation import (
    prepare_data,
    save_preprocessed_data,
    load_raw_data,
)
from src.data_processing.feature_engineering import (
    create_binary_classification,
    select_features,
    save_selected_features,
)
from src.io_models.inputs import Inputs, Model
from src.io_models.outputs import Outputs
from src.models.cnn_lstm_model import build_cnn_lstm
from src.models.lstnet_model import build_lstnet
from src.models.predict import predict
from src.models.tcn_model import build_tcn
from src.models.train import train_model
from src.utils.project_functions import reset_random_seeds

MODEL_MAP: dict[Model, Callable[[tuple[int, int]], Sequential]] = {
    "cnn_lstm": build_cnn_lstm,
    "lstnet": build_lstnet,
    "tcn": build_tcn,
}


def main(inputs: Inputs) -> Outputs:
    # Data cleaning and preprocessing
    logger.info(f"Loading data from {inputs.raw_data_dir_path}...")
    df = load_raw_data(inputs.raw_data_dir_path)
    logger.info("Data loaded. Starting preprocessing...")
    cleaned_df = preprocess_data(df)
    os.makedirs(os.path.dirname(inputs.cleaned_data_path), exist_ok=True)
    cleaned_df.write_parquet(inputs.cleaned_data_path)
    logger.info(f"Cleaned data saved to {inputs.cleaned_data_path}")

    # Feature engineering
    price_df = create_binary_classification(cleaned_df)
    if isinstance(inputs.feature_select, bool):
        features_selected = select_features(cleaned_df, price_df["y"])
    else:
        features_selected = inputs.feature_select

    logger.info(f"Features selected: {', '.join(features_selected)}")
    selected_df = save_selected_features(
        cleaned_df, features_selected, inputs.boruta_data_path
    )

    # Data preparation
    model_train_data = prepare_data(selected_df, price_df, inputs.timesteps)
    save_preprocessed_data(inputs.trained_data_path, model_train_data)
    logger.info("Prepared data")

    # Model training
    input_shape = (inputs.timesteps, model_train_data.x_train.shape[2])

    reset_random_seeds()
    build_model = MODEL_MAP[inputs.model]
    built_model = build_model(input_shape)
    logger.info("Built model")
    train_model(
        built_model,
        model_train_data.x_train,
        model_train_data.y_train,
        inputs.epochs,
        inputs.batch_size,
        inputs.model_save_path,
    )
    logger.info("Trained model")

    # Prediction
    outputs = predict(inputs.model_save_path, inputs.trained_data_path)
    logger.info("Predicted outputs")
    return outputs


if __name__ == "__main__":
    outputs = main(Inputs(model="cnn_lstm", epochs=10))
