"""
SatisfactionPredictor – loads a trained model and makes predictions.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class SatisfactionPredictor:
    """Load a persisted model and predict satisfaction scores."""

    def __init__(self, model_name: str = "gradient_boosting") -> None:
        model_path = config.MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Run ModelTrainer first."
            )
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.model_name = model_name
        logger.info(f"Loaded model: {model_name}")

    def predict(
        self,
        engagement_score: float | np.ndarray,
        experience_score: float | np.ndarray,
    ) -> np.ndarray:
        """Predict satisfaction score from engagement + experience scores."""
        X = np.column_stack([
            np.atleast_1d(engagement_score),
            np.atleast_1d(experience_score),
        ])
        return self.model.predict(X)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict satisfaction scores for a dataframe that has
        'engagement_score' and 'experience_score' columns.
        Adds 'predicted_satisfaction' column.
        """
        df = df.copy()
        df["predicted_satisfaction"] = self.predict(
            df["engagement_score"].values,
            df["experience_score"].values,
        )
        return df
