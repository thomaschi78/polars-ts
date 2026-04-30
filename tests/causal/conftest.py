from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def causal_df() -> pl.DataFrame:
    """Panel with a single series and a clear intervention effect."""
    rng = np.random.default_rng(42)
    n_pre, n_post = 60, 20
    n = n_pre + n_post
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    y = 100.0 + 0.5 * np.arange(n) + rng.normal(0, 1, n)
    # Add treatment effect in post-period
    y[n_pre:] += 10.0
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": dates,
            "y": y.tolist(),
        }
    )


@pytest.fixture()
def intervention_date() -> date:
    return date(2024, 1, 1) + timedelta(days=60)


@pytest.fixture()
def sc_panel_df() -> pl.DataFrame:
    """Panel with one treated unit and three donor units."""
    rng = np.random.default_rng(42)
    n_pre, n_post = 50, 15
    n = n_pre + n_post
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]

    # Shared trend
    trend = 50.0 + 0.3 * np.arange(n)

    rows: list[dict] = []
    # Treated unit: follows trend + treatment effect after intervention
    treated_y = trend + rng.normal(0, 0.5, n)
    treated_y[n_pre:] += 8.0
    for i in range(n):
        rows.append({"unique_id": "treated", "ds": dates[i], "y": float(treated_y[i])})

    # Donor units: follow trend with slight variations
    for donor_name, offset in [("D1", 2.0), ("D2", -1.0), ("D3", 0.5)]:
        donor_y = trend + offset + rng.normal(0, 0.5, n)
        for i in range(n):
            rows.append({"unique_id": donor_name, "ds": dates[i], "y": float(donor_y[i])})

    return pl.DataFrame(rows)


@pytest.fixture()
def sc_intervention_date() -> date:
    return date(2024, 1, 1) + timedelta(days=50)
