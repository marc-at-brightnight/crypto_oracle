import pytest

from src.io_models.inputs import Inputs
from src.main import main
from tests import TESTS_DATA_DIR


def test_main():
    inputs = Inputs(
        model="cnn_lstm",
        raw_data_path=TESTS_DATA_DIR / "raw" / "bitcoin_data.csv",
        processed_dir_path=TESTS_DATA_DIR / "processed",
        model_save_dir_path=TESTS_DATA_DIR / "model_save",
        epochs=1,
        batch_size=50,
    )
    outputs = main(inputs)

    assert outputs.accuracy == pytest.approx(0.520179, abs=1e-4)
    assert outputs.precision == pytest.approx(0.52017, abs=1e-4)
    assert outputs.auc_roc_score == pytest.approx(0.5, abs=1e-4)
    assert outputs.f1_score == pytest.approx(0.684365, abs=1e-4)
    assert outputs.mcc == pytest.approx(0, abs=1e-4)
    assert outputs.recall == pytest.approx(1, abs=1e-4)
