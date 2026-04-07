"""
Telecom Analytics Dashboard – main Streamlit app.
Run with: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.data.loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.analysis.overview import OverviewAnalysis
from src.analysis.engagement import EngagementAnalysis
from src.analysis.experience import ExperienceAnalysis
from src.analysis.satisfaction import SatisfactionAnalysis
from src.models.trainer import ModelTrainer

from dashboard.components import (
    overview_tab,
    engagement_tab,
    experience_tab,
    satisfaction_tab,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📡 Telecom Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Engagement", "🔬 Experience", "⭐ Satisfaction", "⚙️ Pipeline"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data:** XDR/CDR Telecom Records
**Model:** Gradient Boosting Regressor
**MLOps:** MLflow Tracking
""")

# ── Data Loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & cleaning data…", ttl=3600)
def load_data():
    loader = DataLoader()
    df = loader.load_or_create_cleaned()
    return df


@st.cache_data(show_spinner="Engineering features…", ttl=3600)
def build_features(_df):
    fe = FeatureEngineer(_df)
    user_ov  = fe.user_overview_features()
    user_eng = fe.user_engagement_features()
    user_app = fe.app_traffic_features()
    user_exp = fe.user_experience_features()
    return user_ov, user_eng, user_app, user_exp


@st.cache_resource(show_spinner="Building analysis objects…")
def build_analyses(_df, user_ov, user_eng, user_app, user_exp):
    ov_analysis  = OverviewAnalysis(_df, user_ov)
    eng_analysis = EngagementAnalysis(user_eng, user_app)
    exp_analysis = ExperienceAnalysis(user_exp)

    # Pre-run clustering so scores can be computed
    eng_analysis.run_kmeans(k=3)
    exp_analysis.run_kmeans(k=3)

    sat_analysis = SatisfactionAnalysis(eng_analysis.eng, exp_analysis.exp)
    return ov_analysis, eng_analysis, exp_analysis, sat_analysis

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Initialising pipeline…"):
    df = load_data()
    user_ov, user_eng, user_app, user_exp = build_features(df)
    ov_analysis, eng_analysis, exp_analysis, sat_analysis = build_analyses(
        df, user_ov, user_eng, user_app, user_exp
    )

# ── Key Metrics Banner ────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Records",  f"{len(df):,}")
m2.metric("Unique Users",   f"{df[config.USER_ID_COL].nunique():,}")
m3.metric("Handset Types",  f"{df[config.HANDSET_TYPE_COL].nunique():,}")
m4.metric("Total DL (TB)",  f"{df[config.TOTAL_DL_COL].sum()/1e12:.2f}")
m5.metric("Total UL (TB)",  f"{df[config.TOTAL_UL_COL].sum()/1e12:.2f}")
st.markdown("---")

# ── Page Routing ──────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    overview_tab.render(ov_analysis, df)

elif page == "📊 Engagement":
    engagement_tab.render(eng_analysis)

elif page == "🔬 Experience":
    experience_tab.render(exp_analysis)

elif page == "⭐ Satisfaction":
    # Try to load cached model results
    model_results = st.session_state.get("model_results", None)
    satisfaction_tab.render(sat_analysis, model_results)

elif page == "⚙️ Pipeline":
    st.header("⚙️ Run Full Pipeline")
    st.markdown("""
    Use this page to re-run the full analysis pipeline and train the satisfaction model.
    All steps are tracked with **MLflow**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload & Reprocess Data", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared – reload the page to re-run the pipeline.")

    with col2:
        if st.button("🚀 Train Satisfaction Model", type="primary"):
            with st.spinner("Training models with MLflow tracking…"):
                try:
                    trainer = ModelTrainer(sat_analysis.satisfaction_table)
                    results_df = trainer.train_all()
                    best = results_df.iloc[0].to_dict()
                    st.session_state["model_results"] = best
                    st.success(f"Best model: **{best['model_name']}** "
                               f"RMSE={best['rmse']:.4f} R²={best['r2']:.4f}")
                    st.dataframe(results_df[["model_name", "rmse", "mae", "r2", "cv_rmse"]])
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.subheader("MLflow Tracking")
    st.info(f"""
    MLflow tracking URI: `{config.MLFLOW_TRACKING_URI}`
    Experiment: `{config.MLFLOW_EXPERIMENT_NAME}`
    Start the MLflow UI: `mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}`
    """)
