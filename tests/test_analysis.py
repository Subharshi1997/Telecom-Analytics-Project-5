"""
Unit tests for analysis modules – overview, engagement, experience, satisfaction.
"""
import numpy as np
import pandas as pd
import pytest


class TestOverviewAnalysis:
    @pytest.fixture
    def ov_analysis(self, cleaned_df, user_overview):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from src.analysis.overview import OverviewAnalysis
        return OverviewAnalysis(cleaned_df, user_overview)

    def test_top_handsets_count(self, ov_analysis):
        result = ov_analysis.top_handsets(10)
        assert len(result) <= 10
        assert "Handset Type" in result.columns

    def test_top_manufacturers_count(self, ov_analysis):
        result = ov_analysis.top_manufacturers(3)
        assert len(result) <= 3

    def test_top_handsets_per_manufacturer(self, ov_analysis):
        result = ov_analysis.top_handsets_per_manufacturer(3, 5)
        assert isinstance(result, dict)
        assert len(result) <= 3

    def test_dispersion_analysis_columns(self, ov_analysis):
        result = ov_analysis.dispersion_analysis()
        for col in ["Q1", "Q3", "IQR", "Range", "CV_%"]:
            assert col in result.columns

    def test_decile_segmentation(self, ov_analysis):
        result = ov_analysis.decile_segmentation()
        assert "decile" in result.columns
        assert "total_data_bytes_sum" in result.columns
        assert len(result) <= 5

    def test_pca_explained_variance(self, ov_analysis):
        result = ov_analysis.pca_analysis(n_components=2, save=False)
        evr = result["explained_variance_ratio"]
        assert sum(evr) <= 1.0 + 1e-9  # sum ≤ 1
        assert all(v >= 0 for v in evr)


class TestEngagementAnalysis:
    @pytest.fixture
    def eng_analysis(self, user_engagement, user_app_traffic):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from src.analysis.engagement import EngagementAnalysis
        return EngagementAnalysis(user_engagement, user_app_traffic)

    def test_top10_per_metric(self, eng_analysis):
        result = eng_analysis.top10_per_metric()
        assert len(result) >= 1
        for metric, df in result.items():
            assert len(df) <= 10

    def test_kmeans_adds_cluster_col(self, eng_analysis):
        eng_analysis.run_kmeans(k=3)
        assert "engagement_cluster" in eng_analysis.eng.columns

    def test_cluster_statistics_shape(self, eng_analysis):
        eng_analysis.run_kmeans(k=3)
        stats = eng_analysis.cluster_statistics()
        assert len(stats) == 3

    def test_elbow_returns_optimal_k(self, eng_analysis):
        result = eng_analysis.elbow_method(max_k=5, save=False)
        assert "optimal_k" in result
        assert 1 <= result["optimal_k"] <= 5

    def test_top10_per_app(self, eng_analysis):
        result = eng_analysis.top10_per_app()
        for app, df in result.items():
            assert len(df) <= 10


class TestExperienceAnalysis:
    @pytest.fixture
    def exp_analysis(self, user_experience):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from src.analysis.experience import ExperienceAnalysis
        return ExperienceAnalysis(user_experience)

    def test_top_bottom_frequent(self, exp_analysis):
        result = exp_analysis.top_bottom_frequent("avg_tcp_retransmission", n=5)
        assert "top" in result
        assert "bottom" in result
        assert "most_freq" in result
        assert len(result["top"]) <= 5

    def test_kmeans_adds_cluster_col(self, exp_analysis):
        exp_analysis.run_kmeans(k=3)
        assert "experience_cluster" in exp_analysis.exp.columns

    def test_cluster_summary_shape(self, exp_analysis):
        exp_analysis.run_kmeans(k=3)
        summary = exp_analysis.cluster_summary()
        assert len(summary) == 3


class TestSatisfactionAnalysis:
    @pytest.fixture
    def sat_analysis(self, user_engagement, user_experience):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from src.analysis.engagement import EngagementAnalysis
        from src.analysis.experience import ExperienceAnalysis
        from src.analysis.satisfaction import SatisfactionAnalysis
        from src.data.feature_engineering import FeatureEngineer

        eng = EngagementAnalysis(
            user_engagement,
            pd.DataFrame({"MSISDN/Number": user_engagement["MSISDN/Number"]}),
        )
        eng.run_kmeans(k=3)

        exp = ExperienceAnalysis(user_experience)
        exp.run_kmeans(k=3)

        return SatisfactionAnalysis(eng.eng, exp.exp)

    def test_satisfaction_table_columns(self, sat_analysis):
        table = sat_analysis.build_satisfaction_table()
        for col in ["MSISDN/Number", "engagement_score",
                    "experience_score", "satisfaction_score"]:
            assert col in table.columns

    def test_scores_non_negative(self, sat_analysis):
        table = sat_analysis.build_satisfaction_table()
        assert (table["engagement_score"] >= 0).all()
        assert (table["experience_score"] >= 0).all()
        assert (table["satisfaction_score"] >= 0).all()

    def test_top10_satisfied_count(self, sat_analysis):
        top10 = sat_analysis.top10_satisfied()
        assert len(top10) <= 10

    def test_regression_model_metrics(self, sat_analysis):
        results = sat_analysis.train_satisfaction_model()
        assert "rmse" in results
        assert "r2"   in results
        assert results["rmse"] >= 0

    def test_kmeans_on_scores(self, sat_analysis):
        result = sat_analysis.kmeans_on_scores(k=2)
        assert "satisfaction_cluster" in result.columns
        assert result["satisfaction_cluster"].nunique() <= 2
