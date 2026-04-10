"""
FeatureEngineer – builds per-user aggregate features for all four tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class FeatureEngineer:
    """Compute all per-user aggregate feature sets."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._uid = config.USER_ID_COL

    # ── Task 1 – Overview ─────────────────────────────────────────────────────

    def user_overview_features(self) -> pd.DataFrame:
        """
        Aggregate per MSISDN:
          sessions_count, total_duration_ms, total_dl_bytes,
          total_ul_bytes, total_data_bytes + per-app totals
        """
        logger.info("Engineering user overview features…")
        agg: dict = {
            "Dur. (ms)": ["count", "sum"],
            "Total DL (Bytes)": "sum",
            "Total UL (Bytes)": "sum",
            "Total Data (Bytes)": "sum",
        }
        for app, cols in config.APP_COLS.items():
            for col in cols:
                if col in self.df.columns:
                    agg[col] = "sum"

        result = self.df.groupby(self._uid).agg(agg)
        result.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                          for c in result.columns]
        result = result.rename(columns={
            "Dur. (ms)_count": "sessions_count",
            "Dur. (ms)_sum": "total_duration_ms",
            "Total DL (Bytes)_sum": "total_dl_bytes",
            "Total UL (Bytes)_sum": "total_ul_bytes",
            "Total Data (Bytes)_sum": "total_data_bytes",
        })
        result = result.reset_index()

        # Add handset info (last seen per user)
        handset_info = (
            self.df.groupby(self._uid)[[config.HANDSET_MFR_COL, config.HANDSET_TYPE_COL]]
            .last()
            .reset_index()
        )
        result = result.merge(handset_info, on=self._uid, how="left")
        logger.success(f"Overview features: {result.shape}")
        return result

    # ── Task 2 – Engagement ───────────────────────────────────────────────────

    def user_engagement_features(self) -> pd.DataFrame:
        """
        Aggregate per MSISDN:
          sessions_frequency, total_duration_ms, total_traffic_bytes
        """
        logger.info("Engineering user engagement features…")
        agg = {
            "Dur. (ms)": ["count", "sum"],
            "Total DL (Bytes)": "sum",
            "Total UL (Bytes)": "sum",
        }
        result = self.df.groupby(self._uid).agg(agg)
        result.columns = ["_".join(c) for c in result.columns]
        result = result.rename(columns={
            "Dur. (ms)_count": "sessions_frequency",
            "Dur. (ms)_sum": "total_duration_ms",
            "Total DL (Bytes)_sum": "total_dl_bytes",
            "Total UL (Bytes)_sum": "total_ul_bytes",
        })
        result["total_traffic_bytes"] = (
            result["total_dl_bytes"] + result["total_ul_bytes"]
        )
        result = result.reset_index()
        logger.success(f"Engagement features: {result.shape}")
        return result

    def app_traffic_features(self) -> pd.DataFrame:
        """Total traffic per user per application."""
        logger.info("Engineering per-app traffic features…")
        app_agg = {}
        for app, cols in config.APP_COLS.items():
            available = [c for c in cols if c in self.df.columns]
            if available:
                app_agg[f"{app}_total_bytes"] = self.df[available].sum(axis=1)

        app_df = pd.DataFrame(app_agg)
        app_df[self._uid] = self.df[self._uid].values
        result = app_df.groupby(self._uid).sum().reset_index()
        logger.success(f"App traffic features: {result.shape}")
        return result

    # ── Task 3 – Experience ───────────────────────────────────────────────────

    def user_experience_features(self) -> pd.DataFrame:
        """
        Aggregate per MSISDN:
          avg_tcp_retransmission, avg_rtt, avg_throughput, handset_type
        """
        logger.info("Engineering user experience features…")
        numeric_agg = {}
        for col in config.TCP_COLS + config.RTT_COLS + config.THROUGHPUT_COLS:
            if col in self.df.columns:
                numeric_agg[col] = "mean"

        result = self.df.groupby(self._uid).agg(numeric_agg)
        result = result.rename(columns={
            "TCP DL Retrans. Vol (Bytes)": "avg_tcp_dl_retrans",
            "TCP UL Retrans. Vol (Bytes)": "avg_tcp_ul_retrans",
            "Avg RTT DL (ms)": "avg_rtt_dl_ms",
            "Avg RTT UL (ms)": "avg_rtt_ul_ms",
            "Avg Bearer TP DL (kbps)": "avg_throughput_dl_kbps",
            "Avg Bearer TP UL (kbps)": "avg_throughput_ul_kbps",
        })

        # Derived combined metrics
        result["avg_tcp_retransmission"] = (
            result.get("avg_tcp_dl_retrans", 0) + result.get("avg_tcp_ul_retrans", 0)
        ) / 2
        result["avg_rtt_ms"] = (
            result.get("avg_rtt_dl_ms", 0) + result.get("avg_rtt_ul_ms", 0)
        ) / 2
        result["avg_throughput_kbps"] = (
            result.get("avg_throughput_dl_kbps", 0)
            + result.get("avg_throughput_ul_kbps", 0)
        ) / 2

        # Handset type (most frequent per user)
        handset = (
            self.df.groupby(self._uid)[config.HANDSET_TYPE_COL]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown")
            .reset_index()
        )
        result = result.reset_index().merge(handset, on=self._uid, how="left")
        logger.success(f"Experience features: {result.shape}")
        return result
