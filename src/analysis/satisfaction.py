"""
Task 4 – User Satisfaction Analysis
Covers: engagement/experience scores (Euclidean distance), satisfaction score,
top-10 satisfied users, regression model, k-means (k=2), cluster aggregation,
MySQL export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class SatisfactionAnalysis:
    """All Task-4 analyses."""

    def __init__(
        self,
        engagement_df: pd.DataFrame,
        experience_df: pd.DataFrame,
    ) -> None:
        """
        Parameters
        ----------
        engagement_df : must contain `engagement_cluster` and engagement metrics.
        experience_df : must contain `experience_cluster` and experience metrics.
        """
        self.eng = engagement_df.copy()
        self.exp = experience_df.copy()
        self._uid = config.USER_ID_COL
        self._result: pd.DataFrame | None = None

    # ── 4.1  Engagement & Experience Scores ──────────────────────────────────

    def compute_engagement_score(self) -> pd.Series:
        """
        Euclidean distance from each user to the least-engaged cluster centroid.
        Requires `engagement_cluster` to exist in self.eng.
        """
        if "engagement_cluster" not in self.eng.columns:
            raise ValueError("Run EngagementAnalysis.run_kmeans() first.")

        metrics = ["sessions_frequency", "total_duration_ms", "total_traffic_bytes"]
        available = [m for m in metrics if m in self.eng.columns]
        data = self.eng[available].fillna(0).values

        # Least engaged = cluster with smallest mean total_traffic
        traffic_col = "total_traffic_bytes" if "total_traffic_bytes" in self.eng.columns else available[0]
        least_engaged_cluster = (
            self.eng.groupby("engagement_cluster")[traffic_col].mean().idxmin()
        )
        centroid = (
            self.eng[self.eng["engagement_cluster"] == least_engaged_cluster][available]
            .mean()
            .values
        )

        scores = np.linalg.norm(data - centroid, axis=1)
        return pd.Series(scores, index=self.eng.index, name="engagement_score")

    def compute_experience_score(self) -> pd.Series:
        """
        Euclidean distance from each user to the worst-experience cluster centroid.
        Requires `experience_cluster` to exist in self.exp.
        """
        if "experience_cluster" not in self.exp.columns:
            raise ValueError("Run ExperienceAnalysis.run_kmeans() first.")

        exp_metrics = ["avg_tcp_retransmission", "avg_rtt_ms", "avg_throughput_kbps"]
        available = [m for m in exp_metrics if m in self.exp.columns]
        data = self.exp[available].fillna(0).values

        # Worst experience = highest avg TCP retrans
        tcp_col = "avg_tcp_retransmission" if "avg_tcp_retransmission" in self.exp.columns else available[0]
        worst_cluster = (
            self.exp.groupby("experience_cluster")[tcp_col].mean().idxmax()
        )
        centroid = (
            self.exp[self.exp["experience_cluster"] == worst_cluster][available]
            .mean()
            .values
        )

        scores = np.linalg.norm(data - centroid, axis=1)
        return pd.Series(scores, index=self.exp.index, name="experience_score")

    # ── 4.2  Satisfaction Score & Top-10 ─────────────────────────────────────

    def build_satisfaction_table(self) -> pd.DataFrame:
        """
        Merge engagement & experience data, compute scores, and build
        the full satisfaction table per user.
        """
        eng_scores = self.compute_engagement_score()
        self.eng["engagement_score"] = eng_scores

        exp_scores = self.compute_experience_score()
        self.exp["experience_score"] = exp_scores

        # Merge on MSISDN
        merged = pd.merge(
            self.eng[[self._uid, "engagement_score"]],
            self.exp[[self._uid, "experience_score"]],
            on=self._uid,
            how="inner",
        )
        merged["satisfaction_score"] = (
            merged["engagement_score"] + merged["experience_score"]
        ) / 2

        self._result = merged
        logger.success(f"Satisfaction table built: {merged.shape}")
        return merged

    def top10_satisfied(self) -> pd.DataFrame:
        """Top-10 users by satisfaction score."""
        if self._result is None:
            self.build_satisfaction_table()
        return (
            self._result[[self._uid, "engagement_score",
                          "experience_score", "satisfaction_score"]]
            .sort_values("satisfaction_score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

    # ── 4.3  Regression Model ─────────────────────────────────────────────────

    def train_satisfaction_model(self) -> dict:
        """
        Train a Gradient Boosting regressor to predict satisfaction score.
        Returns metrics and the trained model.
        """
        if self._result is None:
            self.build_satisfaction_table()

        features = ["engagement_score", "experience_score"]
        X = self._result[features].values
        y = self._result["satisfaction_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE
        )

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=config.RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        self._satisfaction_model = model
        result = {"mse": mse, "rmse": rmse, "r2": r2, "model": model}
        logger.success(f"Satisfaction model – RMSE: {rmse:.4f} | R²: {r2:.4f}")
        return result

    # ── 4.4  K-Means (k=2) on Scores ─────────────────────────────────────────

    def kmeans_on_scores(self, k: int = config.SATISFACTION_K) -> pd.DataFrame:
        """K-means clustering on engagement & experience scores."""
        if self._result is None:
            self.build_satisfaction_table()

        X = self._result[["engagement_score", "experience_score"]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        self._result["satisfaction_cluster"] = km.fit_predict(X_scaled)
        logger.info(f"Satisfaction K-Means (k={k}) applied.")
        return self._result

    # ── 4.5  Cluster Aggregation ──────────────────────────────────────────────

    def cluster_aggregation(self) -> pd.DataFrame:
        """Average satisfaction & experience score per satisfaction cluster."""
        if self._result is None or "satisfaction_cluster" not in self._result.columns:
            self.kmeans_on_scores()
        return (
            self._result
            .groupby("satisfaction_cluster")[
                ["engagement_score", "experience_score", "satisfaction_score"]
            ]
            .mean()
            .reset_index()
        )

    # ── 4.6  MySQL Export ─────────────────────────────────────────────────────

    def export_to_mysql(self) -> bool:
        """Export satisfaction table to MySQL. Returns True on success."""
        if self._result is None:
            self.build_satisfaction_table()
        from src.database.mysql_connector import MySQLConnector
        connector = MySQLConnector()
        return connector.export_dataframe(
            self._result, table_name="user_satisfaction"
        )

    # ── Getter ────────────────────────────────────────────────────────────────

    @property
    def satisfaction_table(self) -> pd.DataFrame:
        if self._result is None:
            self.build_satisfaction_table()
        return self._result
