"""CuratorAgent: LLM-guided data diagnostics and targeted preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from polars_ts.agents._protocol import LLMBackend, RuleBasedBackend


@dataclass
class CurationReport:
    """Results of data diagnostics."""

    n_observations: int
    n_series: int
    n_missing: int
    n_outliers: int
    detected_period: int | None
    has_trend: bool
    summary: str


class CuratorAgent:
    """Diagnoses data quality issues and applies targeted preprocessing.

    Parameters
    ----------
    backend
        LLM backend for guided diagnostics. Defaults to rule-based heuristics.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with target values.
    outlier_threshold
        Z-score threshold for outlier detection.

    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        outlier_threshold: float = 3.0,
    ) -> None:
        self.backend = backend or RuleBasedBackend()
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.outlier_threshold = outlier_threshold

    def curate(self, df: pl.DataFrame) -> CurationReport:
        """Run diagnostics on the input data and return a report."""
        n_obs = len(df)
        n_missing = df[self.target_col].null_count() + df[self.target_col].is_nan().sum()

        if self.id_col in df.columns:
            n_series = df[self.id_col].n_unique()
        else:
            n_series = 1

        # Outlier detection via z-score per series
        n_outliers = self._count_outliers(df)

        # Seasonality detection via autocorrelation peak
        detected_period = self._detect_period(df)

        # Trend detection via linear regression sign
        has_trend = self._detect_trend(df)

        summary = f"{n_series} series, {n_obs} obs, " f"{n_missing} missing, {n_outliers} outliers"
        if detected_period:
            summary += f", period={detected_period}"
        if has_trend:
            summary += ", trend detected"

        return CurationReport(
            n_observations=n_obs,
            n_series=n_series,
            n_missing=n_missing,
            n_outliers=n_outliers,
            detected_period=detected_period,
            has_trend=has_trend,
            summary=summary,
        )

    def curate_and_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run diagnostics then apply imputation and outlier treatment."""
        result = df

        # Impute missing values with forward fill then backward fill
        if self.id_col in result.columns:
            result = result.sort(self.id_col, self.time_col)
            result = result.with_columns(
                pl.col(self.target_col).fill_nan(None).forward_fill().over(self.id_col).alias(self.target_col)
            )
            result = result.with_columns(
                pl.col(self.target_col).backward_fill().over(self.id_col).alias(self.target_col)
            )
        else:
            result = result.sort(self.time_col)
            result = result.with_columns(pl.col(self.target_col).fill_nan(None).forward_fill().alias(self.target_col))
            result = result.with_columns(pl.col(self.target_col).backward_fill().alias(self.target_col))

        # Clip outliers to +-threshold * std per series
        result = self._clip_outliers(result)

        return result

    def _count_outliers(self, df: pl.DataFrame) -> int:
        total = 0
        if self.id_col in df.columns:
            for _, group_df in df.group_by(self.id_col, maintain_order=True):
                total += self._count_outliers_single(group_df)
        else:
            total = self._count_outliers_single(df)
        return total

    def _count_outliers_single(self, df: pl.DataFrame) -> int:
        values = df[self.target_col].drop_nulls().drop_nans().to_numpy()
        if len(values) < 3:
            return 0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0
        z = np.abs((values - mean) / std)
        return int(np.sum(z > self.outlier_threshold))

    def _detect_period(self, df: pl.DataFrame) -> int | None:
        """Detect dominant period via autocorrelation peak."""
        if self.id_col in df.columns:
            first_id = df[self.id_col][0]
            values = df.filter(pl.col(self.id_col) == first_id)[self.target_col].drop_nulls().drop_nans().to_numpy()
        else:
            values = df[self.target_col].drop_nulls().drop_nans().to_numpy()

        if len(values) < 10:
            return None

        values = values - np.mean(values)
        n = len(values)
        # Compute autocorrelation for lags 2..n//2
        var = np.dot(values, values)
        if var == 0:
            return None

        max_lag = min(n // 2, 60)
        best_lag = None
        best_acf = 0.0
        for lag in range(2, max_lag):
            acf_val = np.dot(values[: n - lag], values[lag:]) / var
            if acf_val > best_acf:
                best_acf = acf_val
                best_lag = lag

        if best_acf > 0.3:
            return best_lag
        return None

    def _detect_trend(self, df: pl.DataFrame) -> bool:
        if self.id_col in df.columns:
            first_id = df[self.id_col][0]
            values = df.filter(pl.col(self.id_col) == first_id)[self.target_col].drop_nulls().drop_nans().to_numpy()
        else:
            values = df[self.target_col].drop_nulls().drop_nans().to_numpy()

        if len(values) < 5:
            return False

        x = np.arange(len(values), dtype=np.float64)
        slope = np.polyfit(x, values, 1)[0]
        value_range = np.ptp(values)
        if value_range == 0:
            return False
        # Trend if slope * n accounts for >20% of value range
        return abs(slope * len(values)) / value_range > 0.2

    def _clip_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.id_col in df.columns:
            mean_expr = pl.col(self.target_col).mean().over(self.id_col)
            std_expr = pl.col(self.target_col).std().over(self.id_col)
        else:
            mean_expr = pl.col(self.target_col).mean()
            std_expr = pl.col(self.target_col).std()

        lower = mean_expr - self.outlier_threshold * std_expr
        upper = mean_expr + self.outlier_threshold * std_expr

        return df.with_columns(pl.col(self.target_col).clip(lower, upper).alias(self.target_col))
