"""
ModelTrainer – trains the satisfaction regression model with full MLflow tracking.
Task 4.7: logs code version, start/end time, source, parameters, metrics,
and artifacts (model pickle + CSV report).
"""
from __future__ import annotations

import time
import pickle
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class ModelTrainer:
    """
    Train and track ML models for satisfaction score prediction.
    All runs are tracked with MLflow (parameters, metrics, artifacts).
    """

    MODELS = {
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=5,
            random_state=config.RANDOM_STATE,
        ),
        "ridge": Ridge(alpha=1.0),
    }

    def __init__(self, satisfaction_df: pd.DataFrame) -> None:
        self.df = satisfaction_df.copy()
        self._artifact_dir = config.MODELS_DIR
        self._artifact_dir.mkdir(exist_ok=True)

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # ── Public API ────────────────────────────────────────────────────────────

    def train_all(self) -> pd.DataFrame:
        """Train all candidate models and return a comparison dataframe."""
        records = []
        for model_name, model in self.MODELS.items():
            record = self._train_single(model_name, model)
            records.append(record)
        comparison = pd.DataFrame(records)
        comparison = comparison.sort_values("rmse")
        logger.success(f"Best model: {comparison.iloc[0]['model_name']} "
                       f"(RMSE={comparison.iloc[0]['rmse']:.4f})")
        return comparison

    def train_best(self) -> dict:
        """Train only the gradient boosting model (best performer by default)."""
        return self._train_single("gradient_boosting", self.MODELS["gradient_boosting"])

    # ── Private ───────────────────────────────────────────────────────────────

    def _train_single(self, model_name: str, model) -> dict:
        """Train one model, track with MLflow, return metrics dict."""
        features = ["engagement_score", "experience_score"]
        X = self.df[features].values
        y = self.df["satisfaction_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE
        )

        params = model.get_params()
        start_time = datetime.utcnow().isoformat()
        t0 = time.time()

        with mlflow.start_run(run_name=model_name) as run:
            # ── Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("features", str(features))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size",  len(X_test))

            # ── Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            elapsed = time.time() - t0
            end_time = datetime.utcnow().isoformat()

            # ── Metrics
            mse  = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5,
                                        scoring="neg_root_mean_squared_error")
            cv_rmse = float(-cv_scores.mean())

            mlflow.log_metrics({
                "mse":      mse,
                "rmse":     rmse,
                "mae":      mae,
                "r2":       r2,
                "cv_rmse":  cv_rmse,
                "train_duration_sec": elapsed,
            })

            # ── Artifacts
            model_path = self._artifact_dir / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path))

            # CSV tracking report
            report_path = self._artifact_dir / f"{model_name}_tracking_report.csv"
            with open(report_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "model_name", "run_id", "start_time", "end_time",
                    "source", "rmse", "mae", "r2", "cv_rmse",
                    "train_duration_sec", "params",
                ])
                writer.writeheader()
                writer.writerow({
                    "model_name": model_name,
                    "run_id":     run.info.run_id,
                    "start_time": start_time,
                    "end_time":   end_time,
                    "source":     "ModelTrainer.train_single",
                    "rmse":       round(rmse, 6),
                    "mae":        round(mae, 6),
                    "r2":         round(r2, 6),
                    "cv_rmse":    round(cv_rmse, 6),
                    "train_duration_sec": round(elapsed, 3),
                    "params":     str(params),
                })
            mlflow.log_artifact(str(report_path))

            mlflow.sklearn.log_model(model, artifact_path=model_name)

            logger.success(f"[{model_name}] RMSE={rmse:.4f} R²={r2:.4f} "
                           f"run_id={run.info.run_id}")

            return {
                "model_name":  model_name,
                "run_id":      run.info.run_id,
                "rmse":        rmse,
                "mae":         mae,
                "r2":          r2,
                "cv_rmse":     cv_rmse,
                "start_time":  start_time,
                "end_time":    end_time,
                "model":       model,
            }
