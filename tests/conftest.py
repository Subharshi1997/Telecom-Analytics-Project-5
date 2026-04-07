"""
Shared fixtures for pytest.
All tests use synthetic dataframes to avoid depending on the raw Excel file.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_raw_df() -> pd.DataFrame:
    """Minimal synthetic raw XDR dataframe (100 rows)."""
    rng = np.random.default_rng(42)
    n = 100
    msisdns = rng.integers(1_000_000_000, 9_999_999_999, size=n)
    df = pd.DataFrame({
        "Bearer Id":                 rng.integers(1, 9999, n),
        "Dur. (ms)":                 rng.integers(1000, 3_600_000, n).astype(float),
        "MSISDN/Number":             msisdns,
        "IMSI":                      rng.integers(1e14, 9e14, n),
        "IMEI":                      rng.integers(1e14, 9e14, n),
        "Handset Manufacturer":      rng.choice(["Samsung", "Huawei", "Apple"], n),
        "Handset Type":              rng.choice(["Galaxy A5", "P30 Lite", "iPhone X"], n),
        "Avg RTT DL (ms)":           rng.uniform(5, 200, n),
        "Avg RTT UL (ms)":           rng.uniform(5, 200, n),
        "Avg Bearer TP DL (kbps)":   rng.uniform(100, 50_000, n),
        "Avg Bearer TP UL (kbps)":   rng.uniform(10, 5_000, n),
        "TCP DL Retrans. Vol (Bytes)": rng.uniform(0, 1e6, n),
        "TCP UL Retrans. Vol (Bytes)": rng.uniform(0, 1e6, n),
        "Social Media DL (Bytes)":   rng.integers(0, 10_000_000, n),
        "Social Media UL (Bytes)":   rng.integers(0, 1_000_000, n),
        "Google DL (Bytes)":         rng.integers(0, 5_000_000, n),
        "Google UL (Bytes)":         rng.integers(0, 500_000, n),
        "Email DL (Bytes)":          rng.integers(0, 2_000_000, n),
        "Email UL (Bytes)":          rng.integers(0, 200_000, n),
        "Youtube DL (Bytes)":        rng.integers(0, 50_000_000, n),
        "Youtube UL (Bytes)":        rng.integers(0, 5_000_000, n),
        "Netflix DL (Bytes)":        rng.integers(0, 40_000_000, n),
        "Netflix UL (Bytes)":        rng.integers(0, 2_000_000, n),
        "Gaming DL (Bytes)":         rng.integers(0, 20_000_000, n),
        "Gaming UL (Bytes)":         rng.integers(0, 1_000_000, n),
        "Other DL (Bytes)":          rng.integers(0, 5_000_000, n),
        "Other UL (Bytes)":          rng.integers(0, 500_000, n),
        "Total DL (Bytes)":          rng.integers(1_000_000, 500_000_000, n),
        "Total UL (Bytes)":          rng.integers(100_000, 50_000_000, n),
    })
    # Inject 10% missing values in some numeric columns
    for col in ["Avg RTT DL (ms)", "TCP DL Retrans. Vol (Bytes)"]:
        idx = rng.choice(n, size=n // 10, replace=False)
        df.loc[idx, col] = np.nan
    return df


@pytest.fixture(scope="session")
def cleaned_df(sample_raw_df) -> pd.DataFrame:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.data.cleaner import DataCleaner
    cleaner = DataCleaner()
    return cleaner.clean(sample_raw_df)


@pytest.fixture(scope="session")
def feature_engineer(cleaned_df):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.data.feature_engineering import FeatureEngineer
    return FeatureEngineer(cleaned_df)


@pytest.fixture(scope="session")
def user_overview(feature_engineer):
    return feature_engineer.user_overview_features()


@pytest.fixture(scope="session")
def user_engagement(feature_engineer):
    return feature_engineer.user_engagement_features()


@pytest.fixture(scope="session")
def user_app_traffic(feature_engineer):
    return feature_engineer.app_traffic_features()


@pytest.fixture(scope="session")
def user_experience(feature_engineer):
    return feature_engineer.user_experience_features()
