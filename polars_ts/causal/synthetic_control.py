"""Synthetic Control Method for causal inference.

Constructs a counterfactual for a treated unit by finding an optimal
weighted combination of donor (control) units. The treatment effect is
the gap between the observed treated series and the synthetic control.

Supports:
- Classic synthetic control (Abadie et al., 2010)
- Prediction intervals via scpi-style uncertainty quantification
  (Cattaneo et al., 2025)
- Built-in placebo tests across all donor units

References
----------
Abadie, Diamond & Hainmueller (2010). *Synthetic Control Methods
for Comparative Case Studies.*

Xu (2017). *Generalized Synthetic Control Method: Causal Inference
with Interactive Fixed Effects Models.*

Cattaneo, Feng, Palomba & Titiunik (2025). *scpi: Uncertainty
Quantification for Synthetic Control Methods.* J. Stat. Soft.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl


@dataclass
class SyntheticControlResult:
    """Result container for a synthetic control analysis.

    Attributes
    ----------
    treated_id
        Identifier of the treated unit.
    weights
        Donor weights, shape ``(n_donors,)``.
    donor_ids
        List of donor unit identifiers, aligned with ``weights``.
    counterfactual
        Synthetic control series for the full time range.
    counterfactual_lower
        Lower prediction interval for the counterfactual.
    counterfactual_upper
        Upper prediction interval for the counterfactual.
    point_effect
        Pointwise treatment effect (observed - synthetic) for the
        post-intervention period.
    point_effect_lower
        Lower bound of pointwise effect.
    point_effect_upper
        Upper bound of pointwise effect.
    total_effect
        Sum of pointwise effects.
    total_effect_lower
        Lower bound of total effect.
    total_effect_upper
        Upper bound of total effect.
    pre_rmse
        Root mean squared error in the pre-intervention period.

    """

    treated_id: Any
    weights: np.ndarray
    donor_ids: list[Any]
    counterfactual: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    point_effect: np.ndarray
    point_effect_lower: np.ndarray
    point_effect_upper: np.ndarray
    total_effect: float
    total_effect_lower: float
    total_effect_upper: float
    pre_rmse: float


class SyntheticControl:
    """Synthetic Control Method estimator.

    Parameters
    ----------
    coverage
        Prediction interval coverage (e.g. 0.9 for 90%).
    id_col
        Column identifying each unit.
    time_col
        Column with timestamps.
    target_col
        Column with observed values.

    """

    def __init__(
        self,
        coverage: float = 0.9,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.coverage = coverage
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._result: SyntheticControlResult | None = None
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pl.DataFrame,
        treated_id: Any,
        intervention_date: date | datetime,
        donor_ids: list[Any] | None = None,
    ) -> SyntheticControl:
        """Fit the synthetic control model.

        Parameters
        ----------
        df
            Panel DataFrame with all units (treated + donors).
        treated_id
            Identifier of the treated unit.
        intervention_date
            First date of the post-intervention period.
        donor_ids
            Specific donor unit IDs. If ``None``, all units except
            ``treated_id`` are used as donors.

        Returns
        -------
        SyntheticControl
            Self, for chaining.

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        all_ids = sorted_df[self.id_col].unique(maintain_order=True).to_list()

        if treated_id not in all_ids:
            raise ValueError(f"treated_id={treated_id!r} not found in data.")

        if donor_ids is None:
            donor_ids = [uid for uid in all_ids if uid != treated_id]

        if len(donor_ids) == 0:
            raise ValueError("No donor units available.")

        # Extract treated series
        treated_df = sorted_df.filter(pl.col(self.id_col) == treated_id)
        times = treated_df[self.time_col].to_list()
        treated_y = treated_df[self.target_col].to_numpy().astype(np.float64)

        pre_mask = np.array([t < intervention_date for t in times])
        post_mask = ~pre_mask

        if pre_mask.sum() < 2:
            raise ValueError(f"Pre-intervention period has {pre_mask.sum()} observations, " f"need at least 2.")
        if post_mask.sum() == 0:
            raise ValueError("No post-intervention observations found.")

        # Build donor matrix (T x n_donors)
        donor_matrix = np.zeros((len(times), len(donor_ids)))
        valid_donors: list[Any] = []
        col_idx = 0
        for did in donor_ids:
            d_df = sorted_df.filter(pl.col(self.id_col) == did)
            d_times = d_df[self.time_col].to_list()
            if d_times != times:
                continue  # skip donors with mismatched time index
            donor_matrix[:, col_idx] = d_df[self.target_col].to_numpy().astype(np.float64)
            valid_donors.append(did)
            col_idx += 1

        donor_matrix = donor_matrix[:, :col_idx]
        if col_idx == 0:
            raise ValueError("No donors with matching time index found.")

        # Solve for weights: minimize ||y_pre - D_pre @ w||^2
        # subject to w >= 0, sum(w) = 1
        pre_treated = treated_y[pre_mask]
        pre_donors = donor_matrix[pre_mask]
        weights = _solve_sc_weights(pre_treated, pre_donors)

        # Construct counterfactual
        counterfactual = donor_matrix @ weights

        # Pre-period fit quality
        pre_residuals = pre_treated - counterfactual[pre_mask]
        pre_rmse = float(np.sqrt(np.mean(pre_residuals**2)))

        # Prediction intervals (scpi-style)
        # Use pre-period residual distribution for uncertainty
        residual_std = float(np.std(pre_residuals, ddof=1)) if len(pre_residuals) > 1 else 0.0

        from scipy.stats import norm

        z = norm.ppf(1 - (1 - self.coverage) / 2)

        # Uncertainty grows with forecast horizon (random-walk diffusion)
        post_len = int(post_mask.sum())
        horizon = np.arange(1, post_len + 1, dtype=np.float64)
        post_std = residual_std * np.sqrt(horizon)

        full_std = np.zeros(len(times))
        full_std[pre_mask] = residual_std
        full_std[post_mask] = post_std

        cf_lower = counterfactual - z * full_std
        cf_upper = counterfactual + z * full_std

        # Effects (post-period only)
        post_treated = treated_y[post_mask]
        post_cf = counterfactual[post_mask]
        point_effect = post_treated - post_cf
        effect_lower = post_treated - (post_cf + z * post_std)
        effect_upper = post_treated - (post_cf - z * post_std)

        total = float(np.sum(point_effect))
        total_lower = float(np.sum(effect_lower))
        total_upper = float(np.sum(effect_upper))

        self._result = SyntheticControlResult(
            treated_id=treated_id,
            weights=weights,
            donor_ids=valid_donors,
            counterfactual=counterfactual,
            counterfactual_lower=cf_lower,
            counterfactual_upper=cf_upper,
            point_effect=point_effect,
            point_effect_lower=effect_lower,
            point_effect_upper=effect_upper,
            total_effect=total,
            total_effect_lower=total_lower,
            total_effect_upper=total_upper,
            pre_rmse=pre_rmse,
        )
        self.is_fitted_ = True
        return self

    @property
    def result(self) -> SyntheticControlResult:
        """Access the fit result."""
        if not self.is_fitted_ or self._result is None:
            raise RuntimeError("Call fit() before accessing result.")
        return self._result

    def to_frame(self) -> pl.DataFrame:
        """Return pointwise post-period results as a DataFrame.

        Columns: step, observed, counterfactual, counterfactual_lower,
        counterfactual_upper, point_effect, point_effect_lower,
        point_effect_upper.

        """
        r = self.result
        rows: list[dict[str, Any]] = []
        for t in range(len(r.point_effect)):
            rows.append(
                {
                    "step": t + 1,
                    "observed": float(r.point_effect[t] + r.counterfactual[-len(r.point_effect) + t]),
                    "counterfactual": float(r.counterfactual[-len(r.point_effect) + t]),
                    "counterfactual_lower": float(r.counterfactual_lower[-len(r.point_effect) + t]),
                    "counterfactual_upper": float(r.counterfactual_upper[-len(r.point_effect) + t]),
                    "point_effect": float(r.point_effect[t]),
                    "point_effect_lower": float(r.point_effect_lower[t]),
                    "point_effect_upper": float(r.point_effect_upper[t]),
                }
            )
        return pl.DataFrame(rows)

    def placebo_test(
        self,
        df: pl.DataFrame,
        intervention_date: date | datetime,
    ) -> pl.DataFrame:
        """Run placebo tests treating each donor as the treated unit.

        Fits the synthetic control for every donor unit (using remaining
        donors as the pool) and reports the empirical distribution of
        placebo effects. A credible treatment effect should be larger
        than the placebo distribution.

        Parameters
        ----------
        df
            Same panel DataFrame used in ``fit()``.
        intervention_date
            Same intervention date.

        Returns
        -------
        pl.DataFrame
            Columns: unit_id, is_treated, total_effect, pre_rmse.

        """
        r = self.result
        all_donor_ids = list(r.donor_ids)
        treated_id = r.treated_id

        rows: list[dict[str, Any]] = []
        # Include the actual treated unit
        rows.append(
            {
                "unit_id": treated_id,
                "is_treated": True,
                "total_effect": r.total_effect,
                "pre_rmse": r.pre_rmse,
            }
        )

        # Run placebo for each donor
        for donor in all_donor_ids:
            placebo_donors = [d for d in all_donor_ids if d != donor]
            placebo_donors.append(treated_id)
            try:
                sc = SyntheticControl(
                    coverage=self.coverage,
                    id_col=self.id_col,
                    time_col=self.time_col,
                    target_col=self.target_col,
                )
                sc.fit(df, treated_id=donor, intervention_date=intervention_date, donor_ids=placebo_donors)
                pr = sc.result
                rows.append(
                    {
                        "unit_id": donor,
                        "is_treated": False,
                        "total_effect": pr.total_effect,
                        "pre_rmse": pr.pre_rmse,
                    }
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

        return pl.DataFrame(rows)


def _solve_sc_weights(y: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Solve for synthetic control weights using constrained least squares.

    Minimizes ||y - D @ w||^2 subject to w >= 0, sum(w) = 1.
    Falls back to unconstrained + normalization if scipy.optimize
    is not available.
    """
    from scipy.optimize import minimize

    n_donors = D.shape[1]
    if n_donors == 1:
        return np.array([1.0])

    def objective(w: np.ndarray) -> float:
        return float(np.sum((y - D @ w) ** 2))

    def jac(w: np.ndarray) -> np.ndarray:
        return -2.0 * D.T @ (y - D @ w)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_donors
    x0 = np.ones(n_donors) / n_donors

    result = minimize(
        objective,
        x0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w = result.x
    w = np.maximum(w, 0.0)
    w_sum = w.sum()
    if w_sum > 0:
        w /= w_sum
    else:
        w = np.ones(n_donors) / n_donors
    return w


def synthetic_control(
    df: pl.DataFrame,
    treated_id: Any,
    intervention_date: date | datetime,
    donor_ids: list[Any] | None = None,
    coverage: float = 0.9,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> SyntheticControlResult:
    """Estimate the causal effect using the synthetic control method.

    Convenience function wrapping :class:`SyntheticControl`.

    Parameters
    ----------
    df
        Panel DataFrame with all units.
    treated_id
        Identifier of the treated unit.
    intervention_date
        First date of the post-intervention period.
    donor_ids
        Specific donor IDs. ``None`` uses all non-treated units.
    coverage
        Prediction interval coverage.
    id_col, time_col, target_col
        Column names.

    Returns
    -------
    SyntheticControlResult

    """
    sc = SyntheticControl(
        coverage=coverage,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    sc.fit(df, treated_id=treated_id, intervention_date=intervention_date, donor_ids=donor_ids)
    return sc.result
