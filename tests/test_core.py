import pytest

from src.io_models.inputs import Inputs
from src.main import main
from tests import TESTS_DATA_DIR


def test_main():
    inputs = Inputs(
        model="cnn_lstm",
        raw_data_dir_path=TESTS_DATA_DIR / "raw",
        processed_dir_path=TESTS_DATA_DIR / "processed",
        model_save_dir_path=TESTS_DATA_DIR / "model_save",
        # skipping feature selection
        feature_select=[
            "adjusted_sopr",
            "ohlc_close",
            "ohlc_high",
            "ohlc_low",
            "ohlc_open",
        ],
        epochs=1,
        batch_size=50,
    )

    outputs = main(inputs)

    assert outputs.model_dump() == pytest.approx(
        {
            "accuracy": 0.5208690680388793,
            "precision": 0.5198161975875933,
            "recall": 0.9977949283351709,
            "f1_score": 0.6835347432024169,
            "auc_roc_score": 0.5024604095357564,
            "mcc": 0.03643784695741684,
        }
    )
