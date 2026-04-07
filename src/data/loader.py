"""
DataLoader – reads raw Excel/CSV data and persists to parquet feature store.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class DataLoader:
    """Load raw telecom XDR data from Excel and persist to the feature store."""

    def __init__(
        self,
        raw_path: Path = config.RAW_DATA_PATH,
        field_desc_path: Path = config.FIELD_DESC_PATH,
    ) -> None:
        self.raw_path = raw_path
        self.field_desc_path = field_desc_path

    # ── Public API ────────────────────────────────────────────────────────────

    def load_raw(self) -> pd.DataFrame:
        """Load raw telecom data from Excel file."""
        logger.info(f"Loading raw data from: {self.raw_path}")
        df = pd.read_excel(self.raw_path, engine="openpyxl")
        logger.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def load_field_descriptions(self) -> pd.DataFrame:
        """Load field description reference table."""
        logger.info(f"Loading field descriptions from: {self.field_desc_path}")
        return pd.read_excel(self.field_desc_path, engine="openpyxl")

    def load_or_create_cleaned(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Return cleaned data from the feature store if available,
        otherwise run the full load → clean → save pipeline.
        """
        if config.CLEANED_DATA_PATH.exists() and not force_reload:
            logger.info("Loading cleaned data from feature store (parquet).")
            return pd.read_parquet(config.CLEANED_DATA_PATH)

        from src.data.cleaner import DataCleaner

        raw = self.load_raw()
        cleaner = DataCleaner()
        cleaned = cleaner.clean(raw)
        # Ensure all object columns are uniform str before parquet serialization
        for col in cleaned.select_dtypes(include=["object"]).columns:
            cleaned[col] = cleaned[col].astype(str)
        cleaned.to_parquet(config.CLEANED_DATA_PATH, index=False)
        logger.success(f"Cleaned data saved → {config.CLEANED_DATA_PATH}")
        return cleaned

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def save_feature(df: pd.DataFrame, path: Path) -> None:
        """Persist a dataframe to a parquet file in the feature store."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Feature saved → {path}")

    @staticmethod
    def load_feature(path: Path) -> pd.DataFrame:
        """Load a feature dataframe from parquet."""
        if not path.exists():
            raise FileNotFoundError(f"Feature not found: {path}")
        return pd.read_parquet(path)
