"""Shared anomaly-flagging logic for decomposition functions."""

import polars as pl


def flag_anomalies(
    df: pl.DataFrame,
    id_col: str,
    threshold: float,
) -> pl.DataFrame:
    """Add an ``is_anomaly`` boolean column based on residual magnitude.

    Flags rows where ``|resid| > threshold * std(resid)`` per group.

    Args:
        df: DataFrame that already contains a ``resid`` column.
        id_col: The column to group by.
        threshold: Number of standard deviations above which a residual is anomalous.

    Returns:
        The input DataFrame with an additional ``is_anomaly`` column.

    """
    return df.with_columns(pl.col("resid").abs().gt(pl.col("resid").std().over(id_col) * threshold).alias("is_anomaly"))
