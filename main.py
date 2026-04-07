"""
main.py – Full pipeline runner (CLI entry point).
Runs all four tasks sequentially and saves outputs to the feature store.

Usage:
    python main.py                    # Run all tasks
    python main.py --task overview    # Run specific task
    python main.py --reload           # Force re-read from Excel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data.loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.analysis.overview import OverviewAnalysis
from src.analysis.engagement import EngagementAnalysis
from src.analysis.experience import ExperienceAnalysis
from src.analysis.satisfaction import SatisfactionAnalysis
from src.models.trainer import ModelTrainer


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_data_pipeline(reload: bool = False):
    logger.info("=== DATA PIPELINE ===")
    loader = DataLoader()
    df = loader.load_or_create_cleaned(force_reload=reload)

    fe = FeatureEngineer(df)
    user_ov  = fe.user_overview_features()
    user_eng = fe.user_engagement_features()
    user_app = fe.app_traffic_features()
    user_exp = fe.user_experience_features()

    DataLoader.save_feature(user_ov,  config.USER_OVERVIEW_PATH)
    DataLoader.save_feature(user_eng, config.USER_ENGAGEMENT_PATH)
    DataLoader.save_feature(user_exp, config.USER_EXPERIENCE_PATH)

    return df, user_ov, user_eng, user_app, user_exp


def run_task1(df, user_ov):
    logger.info("=== TASK 1: USER OVERVIEW ===")
    ov = OverviewAnalysis(df, user_ov)
    print("\n--- Top 10 Handsets ---")
    print(ov.top_handsets(10).to_string(index=False))
    print("\n--- Top 3 Manufacturers ---")
    print(ov.top_manufacturers(3).to_string(index=False))
    print("\n--- Top 5 Handsets per Top Manufacturer ---")
    for mfr, df_hs in ov.top_handsets_per_manufacturer().items():
        print(f"\n  {mfr}:")
        print(df_hs.to_string(index=False))
    print("\n--- Variable Descriptions & Data Types ---")
    print(ov.describe_variables().to_string())
    print("\n--- Basic Metrics (mean, median, std, skewness, kurtosis) ---")
    print(ov.basic_metrics().to_string())
    print("\n--- Dispersion Analysis (sample) ---")
    print(ov.dispersion_analysis().head(10).to_string())
    print("\n--- Decile Segmentation ---")
    print(ov.decile_segmentation().to_string(index=False))
    print("\n--- Bivariate: App vs Total Data ---")
    biv = ov.bivariate_app_vs_total(save=True)
    print(biv.to_string(index=False))
    print("\n--- Correlation Matrix ---")
    corr = ov.correlation_matrix(save=True)
    print(corr.to_string())
    print("\n--- PCA Analysis ---")
    pca = ov.pca_analysis(n_components=2, save=True)
    print(f"Explained variance: {pca['explained_variance_ratio']}")
    ov.plot_univariate(save=True)
    return ov


def run_task2(user_eng, user_app):
    logger.info("=== TASK 2: USER ENGAGEMENT ===")
    eng = EngagementAnalysis(user_eng, user_app)
    print("\n--- Top 10 per Metric ---")
    for metric, df in eng.top10_per_metric().items():
        print(f"\n  {metric}:")
        print(df.to_string(index=False))
    eng.run_kmeans(k=3)
    print("\n--- Cluster Statistics ---")
    print(eng.cluster_statistics().to_string(index=False))
    eng.plot_clusters(save=True)
    print("\n--- Top 10 per App ---")
    for app, df in eng.top10_per_app().items():
        print(f"\n  {app}:")
        print(df.to_string(index=False))
    eng.plot_top3_apps(save=True)
    elbow = eng.elbow_method(max_k=10, save=True)
    print(f"\n--- Elbow: Optimal k = {elbow['optimal_k']} ---")
    return eng


def run_task3(user_exp):
    logger.info("=== TASK 3: USER EXPERIENCE ===")
    exp = ExperienceAnalysis(user_exp)
    print("\n--- Top/Bottom/Frequent ---")
    for metric, data in exp.experience_top_bottom_summary().items():
        print(f"\n  {metric} – Top 10:")
        print(data["top"].to_string())
    print("\n--- Throughput per Handset ---")
    print(exp.throughput_per_handset(save=True).to_string(index=False))
    print("\n--- TCP per Handset ---")
    print(exp.tcp_per_handset(save=True).to_string(index=False))
    exp.run_kmeans(k=3)
    print("\n--- Experience Cluster Summary ---")
    print(exp.cluster_summary().to_string(index=False))
    exp.plot_experience_clusters(save=True)
    return exp


def run_task4(eng, exp):
    logger.info("=== TASK 4: USER SATISFACTION ===")
    sat = SatisfactionAnalysis(eng.eng, exp.exp)
    table = sat.build_satisfaction_table()
    print(f"\n--- Satisfaction Table (shape: {table.shape}) ---")
    print(table.head(10).to_string(index=False))
    print("\n--- Top 10 Satisfied Users ---")
    print(sat.top10_satisfied().to_string(index=False))
    model_result = sat.train_satisfaction_model()
    print(f"\n--- Model Results ---")
    print(f"  RMSE: {model_result['rmse']:.4f}")
    print(f"  R²:   {model_result['r2']:.4f}")
    clustered = sat.kmeans_on_scores(k=2)
    print("\n--- Cluster Aggregation ---")
    print(sat.cluster_aggregation().to_string(index=False))
    DataLoader.save_feature(sat.satisfaction_table, config.USER_SATISFACTION_PATH)
    # Attempt MySQL export
    exported = sat.export_to_mysql()
    if exported:
        logger.success("Satisfaction table exported to MySQL.")
    else:
        logger.warning("MySQL export skipped (check connection settings).")
    return sat


def run_model_tracking(sat):
    logger.info("=== MODEL TRACKING (MLflow) ===")
    trainer = ModelTrainer(sat.satisfaction_table)
    results = trainer.train_all()
    print("\n--- Model Comparison ---")
    print(results[["model_name", "rmse", "mae", "r2", "cv_rmse"]].to_string(index=False))


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Telecom Analytics Pipeline")
    parser.add_argument(
        "--task",
        choices=["all", "overview", "engagement", "experience", "satisfaction", "model"],
        default="all",
    )
    parser.add_argument("--reload", action="store_true", help="Force re-read from Excel")
    args = parser.parse_args()

    df, user_ov, user_eng, user_app, user_exp = run_data_pipeline(reload=args.reload)

    if args.task in ("all", "overview"):
        ov = run_task1(df, user_ov)
    else:
        from src.analysis.overview import OverviewAnalysis
        ov = OverviewAnalysis(df, user_ov)

    if args.task in ("all", "engagement"):
        eng = run_task2(user_eng, user_app)
    else:
        from src.analysis.engagement import EngagementAnalysis
        eng = EngagementAnalysis(user_eng, user_app)
        eng.run_kmeans(k=3)

    if args.task in ("all", "experience"):
        exp = run_task3(user_exp)
    else:
        from src.analysis.experience import ExperienceAnalysis
        exp = ExperienceAnalysis(user_exp)
        exp.run_kmeans(k=3)

    if args.task in ("all", "satisfaction"):
        sat = run_task4(eng, exp)

    if args.task in ("all", "model"):
        if args.task == "model":
            from src.analysis.satisfaction import SatisfactionAnalysis
            sat = SatisfactionAnalysis(eng.eng, exp.exp)
            sat.build_satisfaction_table()
        run_model_tracking(sat)

    logger.success("Pipeline complete.")


if __name__ == "__main__":
    main()
