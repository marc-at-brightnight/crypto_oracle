import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import datetime as dt

from src import DATA_DIR


@dataclass
class ModelTrainData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    time_test: np.ndarray
    price_df: pl.DataFrame
    scaler: StandardScaler


def create_sequences(data, timesteps):
    """
    Function creates sequences for time series data
    """
    X = []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i : i + timesteps])
    return np.array(X)


def load_raw_data(raw_data_dir_path: Path) -> pl.DataFrame:
    latest_start = dt.datetime(2008, 1, 1)
    earliest_end = dt.datetime.now()

    dfs = []
    for filename in os.listdir(raw_data_dir_path):
        if "parquet" not in filename:
            continue

        df = pl.read_parquet(raw_data_dir_path / filename)

        if df["timestamp"].min() > latest_start:
            latest_start = df["timestamp"].min()

        if df["timestamp"].max() < earliest_end:
            earliest_end = df["timestamp"].max()

        dfs.append(df)

    timestamp = pd.date_range(latest_start, earliest_end, freq="H")
    output_df = pl.DataFrame({"timestamp": timestamp}).cast(
        {"timestamp": pl.Datetime("us")}
    )
    for df in dfs:
        output_df = output_df.join(df, on="timestamp", how="left")

    if "ohlc_close" not in output_df.columns:
        raise ValueError("Missing 'ohlc_close' column")

    return output_df


def prepare_data(df: pl.DataFrame, price_df, timesteps: int) -> ModelTrainData:
    """
    Function prepares data
    """
    X = df.drop("timestamp")
    y = price_df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    time_train, time_test = train_test_split(
        df["timestamp"].to_numpy(), test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_reshaped = create_sequences(X_train_scaled, timesteps)
    X_test_reshaped = create_sequences(X_test_scaled, timesteps)
    y_train = y_train[timesteps - 1 :]
    y_test = y_test[timesteps - 1 :]
    time_test = time_test[timesteps:]

    return ModelTrainData(
        X_train_reshaped, X_test_reshaped, y_train, y_test, time_test, price_df, scaler
    )


def save_preprocessed_data(filename: Path, model_train_data: ModelTrainData) -> None:
    """
    Function saves preprocessed data
    """
    with open(filename, "wb") as f:
        pickle.dump(model_train_data, f)


def load_preprocessed_data(filename: Path) -> ModelTrainData:
    """
    Function loads preprocessed data
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
