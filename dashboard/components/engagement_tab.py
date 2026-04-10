"""Streamlit component – Task 2: User Engagement."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(engagement_analysis) -> None:
    """Render the User Engagement tab."""
    st.header("📊 Task 2 – User Engagement Analysis")

    # ── Top-10 per metric ─────────────────────────────────────────────────────
    st.subheader("Top 10 Users per Engagement Metric")
    top10 = engagement_analysis.top10_per_metric()
    metric_labels = {
        "sessions_frequency": "Session Frequency",
        "total_duration_ms":  "Total Duration (ms)",
        "total_traffic_bytes": "Total Traffic (Bytes)",
    }
    cols = st.columns(len(top10))
    for col, (metric, df) in zip(cols, top10.items()):
        with col:
            label = metric_labels.get(metric, metric)
            st.markdown(f"**{label}**")
            st.dataframe(df, use_container_width=True)

    # ── K-Means Clusters ──────────────────────────────────────────────────────
    st.subheader("Engagement Clusters (k=3) – Normalized K-Means")
    with st.spinner("Running K-Means clustering…"):
        engagement_analysis.run_kmeans(k=3)
    cluster_stats = engagement_analysis.cluster_statistics()
    st.markdown("**Cluster Statistics (min, max, mean, total — non-normalized)**")
    st.dataframe(cluster_stats.style.format("{:.2f}"), use_container_width=True)

    # Bar charts: mean of each metric per cluster
    st.markdown("**Mean Engagement Metrics per Cluster**")
    mean_cols = [c for c in cluster_stats.columns if c.endswith("_mean")]
    if mean_cols:
        mean_df = cluster_stats[["engagement_cluster"] + mean_cols].copy()
        mean_df.columns = ["Cluster"] + [c.replace("_mean", "") for c in mean_cols]
        mean_melted = mean_df.melt(id_vars="Cluster", var_name="Metric", value_name="Mean Value")
        mean_melted["Cluster"] = mean_melted["Cluster"].astype(str)
        fig_bar = px.bar(
            mean_melted, x="Metric", y="Mean Value", color="Cluster",
            barmode="group",
            title="Mean Engagement Metrics per Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: sessions_frequency vs total_traffic_bytes
    eng_df = engagement_analysis.eng
    if "engagement_cluster" in eng_df.columns:
        fig = px.scatter(
            eng_df,
            x="sessions_frequency",
            y="total_traffic_bytes",
            color=eng_df["engagement_cluster"].astype(str),
            title="Engagement Clusters: Session Frequency vs Total Traffic",
            labels={
                "sessions_frequency": "Session Frequency",
                "total_traffic_bytes": "Total Traffic (Bytes)",
                "color": "Cluster",
            },
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Cluster Interpretation:**
    - **Low Engagement Cluster:** Users with low session frequency, short durations, and minimal traffic. These are occasional/passive users who may benefit from re-engagement promotions.
    - **Medium Engagement Cluster:** The core customer base with regular usage patterns. Stable session frequency and moderate traffic — represent the majority of subscribers.
    - **High Engagement Cluster:** Power users with high session frequency, long durations, and heavy traffic consumption. Prime targets for premium data plans and bandwidth-heavy service offerings.
    """)

    # ── Top-10 per application ────────────────────────────────────────────────
    st.subheader("Top 10 Users per Application")
    top10_app = engagement_analysis.top10_per_app()
    app_names = list(top10_app.keys())
    selected_app = st.selectbox("Select Application", app_names)
    if selected_app:
        app_df = top10_app[selected_app]
        col_name = f"{selected_app}_total_bytes"
        fig_app = px.bar(
            app_df, x=col_name, y=app_df.columns[0],
            orientation="h",
            title=f"Top 10 Users – {selected_app}",
            color=col_name,
            color_continuous_scale="Reds",
        )
        fig_app.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_app, use_container_width=True)

    # ── Top-3 apps ────────────────────────────────────────────────────────────
    st.subheader("Top 3 Most Used Applications")
    app_df_all = engagement_analysis.app
    app_cols = [c for c in app_df_all.columns if c.endswith("_total_bytes")]
    if app_cols:
        totals = {
            c.replace("_total_bytes", ""): app_df_all[c].sum() / 1e9
            for c in app_cols
        }
        sorted_totals = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:3]
        apps, values = zip(*sorted_totals)
        fig_top3 = go.Figure(go.Bar(
            x=list(apps),
            y=list(values),
            marker_color=["#e74c3c", "#3498db", "#2ecc71"],
            text=[f"{v:.2f} GB" for v in values],
            textposition="outside",
        ))
        fig_top3.update_layout(
            title="Top 3 Applications by Total Traffic",
            xaxis_title="Application",
            yaxis_title="Total Traffic (GB)",
        )
        st.plotly_chart(fig_top3, use_container_width=True)
    st.markdown("""
    **Interpretation:**
    - The top 3 applications account for the vast majority of total network traffic, confirming that a small number of apps drive disproportionate bandwidth consumption.
    - Video and gaming applications typically dominate — their high data volumes are driven by continuous streaming and real-time data transfer requirements.
    - These top apps should be prioritised in QoS (Quality of Service) policies and network capacity planning to maintain a satisfactory user experience for the majority of subscribers.
    """)

    # ── Elbow Method ──────────────────────────────────────────────────────────
    st.subheader("Elbow Method – Optimized k")
    with st.spinner("Computing elbow curve…"):
        elbow = engagement_analysis.elbow_method(max_k=10, save=False)
    k_range = list(range(1, len(elbow["inertias"]) + 1))
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=k_range, y=elbow["inertias"],
        mode="lines+markers", name="Inertia",
        line={"color": "steelblue"},
    ))
    fig_elbow.add_vline(
        x=elbow["optimal_k"], line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal k={elbow['optimal_k']}",
    )
    fig_elbow.update_layout(
        title="Elbow Method – Optimal Number of Clusters",
        xaxis_title="k (Number of Clusters)",
        yaxis_title="Inertia (Within-cluster SSE)",
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.info(f"✅ Optimal k = **{elbow['optimal_k']}** based on second-derivative elbow detection.")
    st.markdown(f"""
    **Interpretation:**
    - The elbow curve shows inertia (within-cluster sum of squares) decreasing as k increases. The rate of decrease slows sharply at **k = {elbow['optimal_k']}**, forming the "elbow".
    - Beyond k = {elbow['optimal_k']}**, adding more clusters yields diminishing returns — clusters become too granular to be meaningfully distinct.
    - This confirms that **{elbow['optimal_k']} engagement groups** best balance cluster cohesion and separation for this dataset.
    - Using k=3 for the main analysis aligns well with the natural low/medium/high engagement segmentation observed in the data.
    """)
