"""
Unit tests for model trainer and predictor.
"""
import numpy as np
import pandas as pd
import pytest
import pickle
import tempfile
from pathlib import Path


@pytest.fixture
def satisfaction_df():
    """Synthetic satisfaction dataframe."""
    rng = np.random.default_rng(0)
    n = 200
    eng = rng.uniform(0, 100, n)
    exp = rng.uniform(0, 100, n)
    sat = (eng + exp) / 2 + rng.normal(0, 5, n)
    return pd.DataFrame({
        "MSISDN/Number":   rng.integers(1_000_000, 9_999_999, n),
        "engagement_score": eng,
        "experience_score": exp,
        "satisfaction_score": sat,
    })


class TestModelTrainer:
    def test_train_best_returns_metrics(self, satisfaction_df, tmp_path, monkeypatch):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import config
        # Redirect artifact dir to tmp
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")

        from src.models.trainer import ModelTrainer
        trainer = ModelTrainer(satisfaction_df)
        result = trainer.train_best()

        assert "rmse" in result
        assert "r2"   in result
        assert result["rmse"] >= 0
        assert -1 <= result["r2"] <= 1.01

    def test_model_artifact_saved(self, satisfaction_df, tmp_path, monkeypatch):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import config
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")

        from src.models.trainer import ModelTrainer
        trainer = ModelTrainer(satisfaction_df)
        trainer.train_best()

        model_path = tmp_path / "gradient_boosting.pkl"
        assert model_path.exists(), "Model pkl artifact not saved"

    def test_tracking_report_csv_saved(self, satisfaction_df, tmp_path, monkeypatch):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import config
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")

        from src.models.trainer import ModelTrainer
        trainer = ModelTrainer(satisfaction_df)
        trainer.train_best()

        csv_path = tmp_path / "gradient_boosting_tracking_report.csv"
        assert csv_path.exists(), "Tracking report CSV not saved"
        df = pd.read_csv(csv_path)
        required_cols = ["model_name", "run_id", "start_time", "end_time",
                         "rmse", "r2", "params"]
        for col in required_cols:
            assert col in df.columns


class TestSatisfactionPredictor:
    def test_predict_single_value(self, satisfaction_df, tmp_path, monkeypatch):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import config
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")

        from src.models.trainer import ModelTrainer
        from src.models.predictor import SatisfactionPredictor

        ModelTrainer(satisfaction_df).train_best()
        predictor = SatisfactionPredictor("gradient_boosting")
        pred = predictor.predict(50.0, 30.0)
        assert len(pred) == 1
        assert isinstance(pred[0], float)

    def test_predict_dataframe(self, satisfaction_df, tmp_path, monkeypatch):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1]))
        import config
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")

        from src.models.trainer import ModelTrainer
        from src.models.predictor import SatisfactionPredictor

        ModelTrainer(satisfaction_df).train_best()
        predictor = SatisfactionPredictor("gradient_boosting")
        result = predictor.predict_dataframe(satisfaction_df)
        assert "predicted_satisfaction" in result.columns
        assert len(result) == len(satisfaction_df)
