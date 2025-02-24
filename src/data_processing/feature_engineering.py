from pathlib import Path

import numpy as np
import polars as pl
from boruta import BorutaPy
from loguru import logger
from sklearn.ensemble import RandomForestClassifier


def create_binary_classification(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create binary classification for price movement.
    """
    price_df = pl.DataFrame({"today": df["ohlc_close"]}).with_columns(
        next_day=pl.col("today").shift(-1)
    )
    price_df = price_df.with_columns(
        y=(pl.col("next_day") > pl.col("today")).cast(pl.Int8)
    )
    price_df[-1, "y"] = 0  # account for last data point that's nan
    return price_df


def select_features(data: pl.DataFrame, target: pl.Series) -> list[str]:
    """
    Select relevant features using Boruta feature selection method.
    """
    X = data.drop("timestamp")
    X_np = X.to_numpy()
    y_np = target.to_numpy()

    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=1)
    feat_selector.fit(X_np, y_np)

    features_bool = feat_selector.support_
    features = np.array(X.columns)
    features_selected = features[features_bool]

    return features_selected.tolist()


def save_selected_features(
    df: pl.DataFrame, features_selected: list[str], output_path: Path
) -> pl.DataFrame:
    """
    Save selected features.
    """
    selected_df = df.select("timestamp", *features_selected)
    selected_df.write_parquet(output_path)
    logger.info(f"Selected features saved to {output_path}")
    return selected_df
