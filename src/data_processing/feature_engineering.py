from pathlib import Path

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
import polars as pl

np.int = int
np.float = float
np.bool = bool


def create_binary_classification(df: pl.DataFrame) -> pl.Series:
    """
    Create binary classification for price movement.
    """
    price_df = pl.DataFrame({'today': df['ohlc_close']}).with_columns(next_day=pl.col('today').shift(-1))[:-1]
    return (price_df['next_day'] > price_df['today']).cast(pl.Int8)


def select_features(
    data: pl.DataFrame, target: pl.Series
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select relevant features using Boruta feature selection method.
    """
    X = data[:-1].drop('timestamp')
    X_np = X.to_numpy()
    y_np = target.to_numpy()

    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=1)
    feat_selector.fit(X_np, y_np)

    features_bool = feat_selector.support_
    feature_ranking = feat_selector.ranking_
    features = np.array(X.columns)
    features_selected = features[features_bool]
    features_selected_tentative = features[feature_ranking <= feat_selector.max_iter]

    return features_selected, features_selected_tentative


def save_selected_features(
    df: pl.DataFrame, features_selected: np.ndarray, output_path: Path
) -> None:
    """
    Save selected features.
    """
    features_selected = np.insert(features_selected, 0, "timestamp")
    df.select(features_selected).to_csv(output_path, index=False)
    logger.info(f"Selected features saved to {output_path}")
