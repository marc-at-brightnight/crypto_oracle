from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src import DATA_DIR

Model = Literal["cnn_lstm", "lstnet", "tcn"]


class Inputs(BaseModel):
    model: Model
    raw_data_path: Path = DATA_DIR / "raw" / "bitcoin_data.csv"
    processed_dir_path: Path = DATA_DIR / "processed"
    model_save_dir_path: Path = DATA_DIR / "model_save"
    timesteps: int = 5
    epochs: int = 1000
    batch_size: int = 50

    @property
    def model_save_path(self) -> Path:
        return self.model_save_dir_path / f"{self.model}.keras"

    @property
    def cleaned_data_path(self) -> Path:
        return self.processed_dir_path / "cleaned_data.csv"

    @property
    def boruta_data_path(self) -> Path:
        return self.processed_dir_path / "Boruta_data.csv"

    @property
    def trained_data_path(self) -> Path:
        return self.processed_dir_path / "split_data.pkl"
