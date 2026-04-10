"""
MySQLConnector – handles all database interactions.
Uses SQLAlchemy for ORM-level compatibility + mysql-connector-python for raw queries.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class MySQLConnector:
    """CRUD operations for the telecom analytics MySQL database."""

    def __init__(self) -> None:
        cfg = config.MYSQL_CONFIG
        conn_str = (
            f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        )
        self._engine = create_engine(conn_str, echo=False)
        self._ensure_database()

    # ── Public API ────────────────────────────────────────────────────────────

    def export_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> bool:
        """
        Write a dataframe to a MySQL table.
        Parameters
        ----------
        df         : dataframe to write
        table_name : target table name
        if_exists  : 'replace' | 'append' | 'fail'
        """
        try:
            df.to_sql(table_name, con=self._engine, if_exists=if_exists, index=False)
            logger.success(f"Exported {len(df):,} rows to table `{table_name}`")
            return True
        except Exception as exc:
            logger.error(f"MySQL export failed: {exc}")
            return False

    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read an entire table into a dataframe."""
        with self._engine.connect() as conn:
            return pd.read_sql(f"SELECT * FROM `{table_name}`", conn)

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute a raw SELECT query and return results as a dataframe."""
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_database(self) -> None:
        """Create the database if it doesn't exist (best-effort)."""
        cfg = config.MYSQL_CONFIG
        base_conn_str = (
            f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}"
        )
        try:
            base_engine = create_engine(base_conn_str, echo=False)
            with base_engine.connect() as conn:
                conn.execute(
                    text(f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}`")
                )
            logger.debug(f"Database `{cfg['database']}` ensured.")
        except Exception as exc:
            logger.warning(f"Could not ensure database: {exc}")
