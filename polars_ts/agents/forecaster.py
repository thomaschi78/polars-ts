"""ForecasterAgent: model fitting, validation, and adaptive ensemble selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from polars_ts.agents._protocol import LLMBackend, RuleBasedBackend
from polars_ts.agents.planner import ForecastPlan


@dataclass
class ForecastAgentResult:
    """Output of the ForecasterAgent."""

    predictions: pl.DataFrame
    best_model: str
    model_scores: dict[str, float]
    all_predictions: dict[str, pl.DataFrame] = field(default_factory=dict)


# Map plan candidate names to forecast functions
_MODEL_REGISTRY: dict[str, str] = {
    "naive": "polars_ts.models.baselines.naive_forecast",
    "moving_average": "polars_ts.models.baselines.moving_average_forecast",
    "ses": "polars_ts.models.exponential_smoothing.ses_forecast",
    "holt": "polars_ts.models.exponential_smoothing.holt_forecast",
    "holt_winters": "polars_ts.models.exponential_smoothing.holt_winters_forecast",
}


def _import_model(dotted_path: str) -> Any:
    import importlib

    module_path, _, func_name = dotted_path.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


class ForecasterAgent:
    """Fits candidate models, validates, and selects the best.

    Parameters
    ----------
    backend
        LLM backend for guided decisions. Defaults to rule-based heuristics.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with target values.

    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.backend = backend or RuleBasedBackend()
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def forecast(self, df: pl.DataFrame, plan: ForecastPlan) -> ForecastAgentResult:
        """Fit each candidate model and select the best by validation MAE."""
        h = plan.horizon

        # Split into train / validation
        train, val = self._train_val_split(df, h)

        scores: dict[str, float] = {}
        all_preds: dict[str, pl.DataFrame] = {}

        for name in plan.candidates:
            dotted = _MODEL_REGISTRY.get(name)
            if dotted is None:
                continue
            try:
                fn = _import_model(dotted)
                extra = plan.config.get(name, {})
                preds = fn(train, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col, **extra)
                all_preds[name] = preds
                mae = self._score(val, preds)
                scores[name] = mae
            except Exception:
                scores[name] = float("inf")

        if not scores:
            # Fallback: just use naive
            from polars_ts.models.baselines import naive_forecast

            preds = naive_forecast(train, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col)
            return ForecastAgentResult(
                predictions=preds,
                best_model="naive",
                model_scores={"naive": float("nan")},
                all_predictions={"naive": preds},
            )

        best_name = min(scores, key=lambda k: scores[k])

        # Re-fit best model on full data for final predictions
        best_fn = _import_model(_MODEL_REGISTRY[best_name])
        extra = plan.config.get(best_name, {})
        final_preds = best_fn(df, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col, **extra)

        return ForecastAgentResult(
            predictions=final_preds,
            best_model=best_name,
            model_scores=scores,
            all_predictions=all_preds,
        )

    def _train_val_split(
        self,
        df: pl.DataFrame,
        h: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split each series: last h observations go to validation."""
        has_id = self.id_col in df.columns
        sort_cols = [self.id_col, self.time_col] if has_id else [self.time_col]
        sorted_df = df.sort(*sort_cols)
        if has_id:
            ranked = sorted_df.with_columns(
                pl.col(self.time_col).rank(method="ordinal", descending=True).over(self.id_col).alias("__rank")
            )
        else:
            ranked = sorted_df.with_columns(
                pl.col(self.time_col).rank(method="ordinal", descending=True).alias("__rank")
            )
        train = ranked.filter(pl.col("__rank") > h).drop("__rank")
        val = ranked.filter(pl.col("__rank") <= h).drop("__rank")
        return train, val

    def _score(self, val: pl.DataFrame, preds: pl.DataFrame) -> float:
        """Compute MAE between validation actuals and predictions."""
        join_cols = [c for c in [self.id_col, self.time_col] if c in val.columns and c in preds.columns]
        if not join_cols:
            return float("inf")

        joined = val.join(preds, on=join_cols, how="inner")
        if joined.is_empty():
            return float("inf")

        actual = joined[self.target_col].to_numpy().astype(np.float64)
        predicted = joined["y_hat"].to_numpy().astype(np.float64)
        return _compute_mae(actual, predicted)
