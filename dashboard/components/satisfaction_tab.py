"""Streamlit component – Task 4: User Satisfaction."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(satisfaction_analysis, model_results: dict | None = None) -> None:
    """Render the User Satisfaction tab."""
    st.header("⭐ Task 4 – User Satisfaction Analysis")

    # ── Build satisfaction table ──────────────────────────────────────────────
    with st.spinner("Computing satisfaction scores…"):
        sat_table = satisfaction_analysis.build_satisfaction_table()

    st.subheader("Satisfaction Score Distribution")
    fig_hist = px.histogram(
        sat_table, x="satisfaction_score",
        nbins=50, color_discrete_sequence=["#2ecc71"],
        title="Satisfaction Score Distribution",
        labels={"satisfaction_score": "Satisfaction Score"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Top-10 satisfied users ────────────────────────────────────────────────
    st.subheader("Top 10 Most Satisfied Customers")
    top10 = satisfaction_analysis.top10_satisfied()
    st.dataframe(
        top10.style.background_gradient(subset=["satisfaction_score"], cmap="Greens"),
        use_container_width=True,
    )

    # ── 2D score scatter ──────────────────────────────────────────────────────
    st.subheader("Engagement vs Experience Score Space")
    fig_scatter = px.scatter(
        sat_table,
        x="engagement_score",
        y="experience_score",
        color="satisfaction_score",
        color_continuous_scale="Viridis",
        title="User Score Space (colour = satisfaction)",
        opacity=0.6,
        labels={
            "engagement_score": "Engagement Score",
            "experience_score": "Experience Score",
            "satisfaction_score": "Satisfaction",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── k=2 clustering ────────────────────────────────────────────────────────
    st.subheader("Satisfaction Clusters (k=2)")
    with st.spinner("Clustering satisfaction data…"):
        clustered = satisfaction_analysis.kmeans_on_scores(k=2)
    cluster_agg = satisfaction_analysis.cluster_aggregation()
    st.dataframe(cluster_agg.style.format("{:.4f}"), use_container_width=True)

    if "satisfaction_cluster" in clustered.columns:
        fig_cl = px.scatter(
            clustered,
            x="engagement_score",
            y="experience_score",
            color=clustered["satisfaction_cluster"].astype(str),
            title="Satisfaction Clusters (k=2)",
            color_discrete_sequence=["#e74c3c", "#3498db"],
            opacity=0.6,
        )
        st.plotly_chart(fig_cl, use_container_width=True)

    # ── Regression model results ──────────────────────────────────────────────
    st.subheader("Satisfaction Prediction Model")
    if model_results:
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{model_results.get('rmse', 0):.4f}")
        col2.metric("MAE",  f"{model_results.get('mae', 0):.4f}")
        col3.metric("R²",   f"{model_results.get('r2', 0):.4f}")
    else:
        st.info("Train the model from the **Run Pipeline** page to see metrics.")

    # ── Interactive prediction ────────────────────────────────────────────────
    st.subheader("🔮 Predict Satisfaction Score")
    c1, c2 = st.columns(2)
    eng_score = c1.number_input(
        "Engagement Score", min_value=0.0,
        value=float(sat_table["engagement_score"].mean()),
        step=0.01, format="%.4f",
    )
    exp_score = c2.number_input(
        "Experience Score", min_value=0.0,
        value=float(sat_table["experience_score"].mean()),
        step=0.01, format="%.4f",
    )
    if st.button("Predict"):
        try:
            from src.models.predictor import SatisfactionPredictor
            predictor = SatisfactionPredictor()
            pred = predictor.predict(eng_score, exp_score)[0]
            st.success(f"Predicted Satisfaction Score: **{pred:.4f}**")
        except FileNotFoundError:
            # Fallback: simple average
            pred = (eng_score + exp_score) / 2
            st.warning(f"Model not trained yet – using simple average: **{pred:.4f}**")

    # ── MySQL export status ───────────────────────────────────────────────────
    st.subheader("💾 Database Export")
    if st.button("Export to MySQL"):
        with st.spinner("Exporting…"):
            ok = satisfaction_analysis.export_to_mysql()
        if ok:
            st.success("✅ Data exported to MySQL table `user_satisfaction`")
        else:
            st.error("❌ MySQL export failed – check connection settings in config.py")
