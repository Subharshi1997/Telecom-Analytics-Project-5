"""
DataCleaner – handles missing values and outliers using IQR method.
Strategy:
  - Numeric columns: replace missing & outliers with column mean
  - Categorical columns: replace missing with mode
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class DataCleaner:
    """Clean raw telecom XDR data."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return a clean copy."""
        logger.info("Starting data cleaning pipeline…")
        df = df.copy()
        df = self._fix_column_names(df)
        df = self._safe_parquet_types(df)
        df = self._impute_missing(df)
        df = self._handle_outliers(df)
        df = self._add_total_data(df)
        logger.success(f"Cleaning complete. Shape: {df.shape}")
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from column names."""
        df.columns = [c.strip() for c in df.columns]
        return df

    @staticmethod
    def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace NaN values:
          - Numeric  → column mean
          - Object   → column mode (most frequent value)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            missing = df[col].isna().sum()
            if missing:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                logger.debug(f"[missing] {col}: filled {missing} NaN → mean={mean_val:.4f}")

        for col in object_cols:
            missing = df[col].isna().sum()
            if missing:
                mode_val = df[col].mode()[0] if len(df[col].mode()) else "Unknown"
                df[col] = df[col].fillna(mode_val)
                logger.debug(f"[missing] {col}: filled {missing} NaN → mode='{mode_val}'")

        return df

    @staticmethod
    def _handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """
        IQR-based outlier treatment for numeric columns.
        Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are replaced with the column mean.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Skip identifier-like columns
        skip_cols = {"Bearer Id", "IMSI", "MSISDN/Number", "IMEI",
                     "Start ms", "End ms"}

        for col in numeric_cols:
            if col in skip_cols:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            n_outliers = outlier_mask.sum()
            if n_outliers:
                mean_val = df[col][~outlier_mask].mean()
                df.loc[outlier_mask, col] = mean_val
                logger.debug(f"[outlier] {col}: replaced {n_outliers} → mean={mean_val:.4f}")

        return df

    @staticmethod
    def _add_total_data(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure 'Total Data (Bytes)' column exists as DL + UL."""
        if "Total DL (Bytes)" in df.columns and "Total UL (Bytes)" in df.columns:
            df["Total Data (Bytes)"] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]
        return df

    @staticmethod
    def _safe_parquet_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all integer columns are within PyArrow-safe int64 range.
        Columns with values outside [-2^63, 2^63-1] are cast to string.
        This handles large IMSI / IMEI values from the raw Excel file.
        """
        INT64_MAX = np.iinfo(np.int64).max
        INT64_MIN = np.iinfo(np.int64).min

        for col in df.select_dtypes(include=[np.integer]).columns:
            col_max = df[col].max()
            col_min = df[col].min()
            if col_max > INT64_MAX or col_min < INT64_MIN:
                df[col] = df[col].astype(str)
                logger.debug(f"[parquet-fix] {col}: cast to str (value overflow int64)")
            elif df[col].dtype != np.int64:
                df[col] = df[col].astype(np.int64)
        return df

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary statistics dataframe for all numeric columns."""
        return df.select_dtypes(include=[np.number]).describe().T
