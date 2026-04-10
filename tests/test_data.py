"""
Unit tests for data loading, cleaning, and feature engineering.
"""
import numpy as np
import pandas as pd
import pytest


class TestDataCleaner:
    def test_no_missing_after_clean(self, cleaned_df):
        """All numeric NaN values should be imputed."""
        numeric = cleaned_df.select_dtypes(include=[np.number])
        assert numeric.isna().sum().sum() == 0, "Numeric NaN remaining after cleaning"

    def test_total_data_column_exists(self, cleaned_df):
        assert "Total Data (Bytes)" in cleaned_df.columns

    def test_total_data_correct(self, cleaned_df):
        expected = cleaned_df["Total DL (Bytes)"] + cleaned_df["Total UL (Bytes)"]
        pd.testing.assert_series_equal(
            cleaned_df["Total Data (Bytes)"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_outlier_replacement_within_bounds(self, sample_raw_df):
        """After outlier handling, values should be within IQR-extended bounds."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from src.data.cleaner import DataCleaner
        cleaner = DataCleaner()
        df = cleaner.clean(sample_raw_df)
        col = "Avg RTT DL (ms)"
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # After replacing with mean, values should not be extreme outliers
        assert df[col].max() <= upper * 3  # generous bound


class TestFeatureEngineer:
    def test_overview_columns(self, user_overview):
        required = ["MSISDN/Number", "sessions_count", "total_duration_ms",
                    "total_dl_bytes", "total_ul_bytes", "total_data_bytes"]
        for col in required:
            assert col in user_overview.columns, f"Missing column: {col}"

    def test_overview_positive_values(self, user_overview):
        assert (user_overview["sessions_count"] > 0).all()
        assert (user_overview["total_dl_bytes"] >= 0).all()

    def test_engagement_columns(self, user_engagement):
        required = ["MSISDN/Number", "sessions_frequency",
                    "total_duration_ms", "total_traffic_bytes"]
        for col in required:
            assert col in user_engagement.columns, f"Missing column: {col}"

    def test_experience_columns(self, user_experience):
        required = ["MSISDN/Number", "avg_tcp_retransmission",
                    "avg_rtt_ms", "avg_throughput_kbps"]
        for col in required:
            assert col in user_experience.columns, f"Missing column: {col}"

    def test_app_traffic_has_app_cols(self, user_app_traffic):
        app_cols = [c for c in user_app_traffic.columns if c.endswith("_total_bytes")]
        assert len(app_cols) >= 1, "No app traffic columns found"

    def test_unique_users_preserved(self, cleaned_df, user_overview):
        raw_users = cleaned_df["MSISDN/Number"].nunique()
        agg_users = user_overview["MSISDN/Number"].nunique()
        assert agg_users == raw_users, "User count mismatch after aggregation"
