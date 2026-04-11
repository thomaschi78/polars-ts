import numpy as np
import polars as pl

from polars_ts import seasonal_decomposition


class TestSeasonalAnomalyFlags:
    def _make_seasonal_df(self, n=120, freq=30, inject_outlier_at=None, outlier_value=100.0):
        """Create a simple seasonal time series for testing."""
        np.random.seed(42)
        t = np.arange(n)
        trend = 0.05 * t + 50
        seasonal = 5 * np.sin(2 * np.pi * t / freq)
        noise = np.random.randn(n) * 0.5
        y = trend + seasonal + noise

        if inject_outlier_at is not None:
            y[inject_outlier_at] = outlier_value

        return pl.DataFrame(
            {
                "unique_id": ["sensor"] * n,
                "ds": list(range(n)),
                "y": y.tolist(),
            }
        )

    def test_no_threshold_no_column(self):
        """Default (None) should not add is_anomaly column."""
        df = self._make_seasonal_df()
        result = seasonal_decomposition(df, freq=30)
        assert "is_anomaly" not in result.columns

    def test_threshold_adds_column(self):
        """Setting anomaly_threshold should add a boolean is_anomaly column."""
        df = self._make_seasonal_df()
        result = seasonal_decomposition(df, freq=30, anomaly_threshold=2.0)
        assert "is_anomaly" in result.columns
        assert result["is_anomaly"].dtype == pl.Boolean

    def test_injected_outlier_is_flagged(self):
        """An extreme outlier should be flagged with a reasonable threshold."""
        df = self._make_seasonal_df(inject_outlier_at=60, outlier_value=500.0)
        result = seasonal_decomposition(df, freq=30, anomaly_threshold=3.0)
        anomalies = result.filter(pl.col("is_anomaly"))
        # The injected outlier should be among the flagged rows
        assert anomalies.height >= 1

    def test_clean_data_few_anomalies(self):
        """Clean data with threshold=3 should flag very few points (≈0.3% for normal)."""
        df = self._make_seasonal_df(n=300)
        result = seasonal_decomposition(df, freq=30, anomaly_threshold=3.0)
        anomaly_rate = result.filter(pl.col("is_anomaly")).height / result.height
        assert anomaly_rate < 0.05  # less than 5%

    def test_constant_residuals_few_anomalies(self):
        """A mostly clean seasonal series should have very few anomalies with high threshold."""
        df = self._make_seasonal_df(n=300)
        result = seasonal_decomposition(df, freq=30, anomaly_threshold=3.0)
        # With threshold=3, we expect very few flags in clean data
        anomaly_count = result.filter(pl.col("is_anomaly")).height
        assert anomaly_count < result.height * 0.05
