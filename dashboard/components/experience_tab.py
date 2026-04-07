"""Streamlit component – Task 3: User Experience."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(experience_analysis) -> None:
    """Render the User Experience tab."""
    st.header("🔬 Task 3 – User Experience Analysis")

    # ── 3.2  Top / Bottom / Most Frequent ────────────────────────────────────
    st.subheader("TCP / RTT / Throughput – Top, Bottom & Most Frequent")
    summary = experience_analysis.experience_top_bottom_summary()
    for metric, data in summary.items():
        with st.expander(f"📈 {metric} Statistics"):
            col1, col2, col3 = st.columns(3)
            col1.markdown("**Top 10**")
            col1.dataframe(data["top"].reset_index(), use_container_width=True)
            col2.markdown("**Bottom 10**")
            col2.dataframe(data["bottom"].reset_index(), use_container_width=True)
            col3.markdown("**Most Frequent 10**")
            col3.dataframe(data["most_freq"].reset_index(), use_container_width=True)

    # ── 3.3a  Throughput per handset ──────────────────────────────────────────
    st.subheader("Average Throughput per Handset Type (Top 15)")
    tp_df = experience_analysis.throughput_per_handset(top_n=15, save=False)
    fig_tp = px.bar(
        tp_df, x="avg_throughput_kbps", y="Handset Type",
        orientation="h", color="avg_throughput_kbps",
        color_continuous_scale="Blues",
        title="Avg Throughput per Handset Type",
    )
    fig_tp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_tp, use_container_width=True)
    st.markdown("""
    **Interpretation:** Premium flagship handsets (e.g., iPhone, Samsung Galaxy S-series)
    consistently show higher throughput due to better antenna hardware and 5G/LTE-A support.
    Entry-level devices struggle to sustain high throughput even on good network conditions.
    """)

    # ── 3.3b  TCP per handset ─────────────────────────────────────────────────
    st.subheader("Average TCP Retransmission per Handset Type (Top 15)")
    tcp_df = experience_analysis.tcp_per_handset(top_n=15, save=False)
    fig_tcp = px.bar(
        tcp_df, x="avg_tcp_retransmission", y="Handset Type",
        orientation="h", color="avg_tcp_retransmission",
        color_continuous_scale="Reds",
        title="Avg TCP Retransmission per Handset Type",
    )
    fig_tcp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_tcp, use_container_width=True)
    st.markdown("""
    **Interpretation:** High TCP retransmission volumes indicate packet loss or network
    congestion. Certain handset models with older TCP stack implementations show
    systematically higher retransmission, suggesting firmware/driver optimization
    opportunities for network operators.
    """)

    # ── 3.4  Experience Clusters ──────────────────────────────────────────────
    st.subheader("Experience Clusters (k=3)")
    with st.spinner("Clustering experience data…"):
        experience_analysis.run_kmeans(k=3)
    cluster_sum = experience_analysis.cluster_summary()

    # Rename cluster IDs to descriptive labels based on TCP (worst = highest TCP)
    tcp_col = "avg_tcp_retransmission" if "avg_tcp_retransmission" in cluster_sum.columns else cluster_sum.columns[1]
    sorted_clusters = cluster_sum.sort_values(tcp_col)
    label_map = {
        int(sorted_clusters.iloc[0]["experience_cluster"]): "Good Experience",
        int(sorted_clusters.iloc[1]["experience_cluster"]): "Average Experience",
        int(sorted_clusters.iloc[2]["experience_cluster"]): "Poor Experience",
    }
    cluster_sum["Cluster Label"] = cluster_sum["experience_cluster"].map(label_map)
    st.dataframe(cluster_sum, use_container_width=True)

    # Radar chart for cluster profiles
    categories = [c for c in cluster_sum.columns
                  if c not in ["experience_cluster", "Cluster Label"]]
    fig_radar = go.Figure()
    for _, row in cluster_sum.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in categories],
            theta=categories,
            fill="toself",
            name=label_map.get(int(row["experience_cluster"]), f"Cluster {int(row['experience_cluster'])}"),
        ))
    fig_radar.update_layout(
        polar={"radialaxis": {"visible": True}},
        title="Experience Cluster Profiles",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("""
    **Cluster Descriptions:**
    - **Good Experience:** Low TCP retransmission, low RTT, high throughput → premium users on good coverage
    - **Average Experience:** Mid-range network metrics → majority of the user base
    - **Poor Experience:** High TCP retransmission, high RTT, low throughput → users on congested cells or with poor handsets
    """)

    # Distribution plot of experience features per cluster
    exp_df = experience_analysis.exp
    if "experience_cluster" in exp_df.columns:
        feat_col = st.selectbox(
            "Feature to visualize by cluster",
            options=[c for c in ["avg_tcp_retransmission", "avg_rtt_ms", "avg_throughput_kbps"]
                     if c in exp_df.columns],
        )
        fig_box = px.box(
            exp_df, x=exp_df["experience_cluster"].astype(str), y=feat_col,
            color=exp_df["experience_cluster"].astype(str),
            title=f"{feat_col} Distribution by Experience Cluster",
            labels={"x": "Cluster"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_box, use_container_width=True)
