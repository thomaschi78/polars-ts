//! Sen's slope estimator for Polars Series.
//!
//! Computes the Theil-Sen estimator: the median of all pairwise slopes
//! `(y[j] - y[i]) / (j - i)` for `j > i`. This is a robust, non-parametric
//! estimate of the trend magnitude that is resistant to outliers.
//!
//! Null values are tracked by their original position so that the time gap
//! `(j - i)` reflects the true spacing. NaN values are filtered out.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// Compute the Sen's slope estimator on the first Series in `inputs`.
///
/// # Parameters
/// - `inputs`: a slice of Series, with the first entry expected to be a float Series (f64).
///
/// # Returns
/// A single-valued Float64 Series containing the median pairwise slope.
/// For fewer than 2 valid (non-null, non-NaN) values, returns `0.0`.
#[polars_expr(output_type=Float64)]
pub fn sens_slope(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.f64()?;

    // Collect (original_index, value) pairs, skipping nulls and NaNs
    let indexed_vals: Vec<(usize, f64)> = ca
        .into_iter()
        .enumerate()
        .filter_map(|(i, opt_v)| {
            opt_v.and_then(|v| if v.is_nan() { None } else { Some((i, v)) })
        })
        .collect();

    let n = indexed_vals.len();
    if n < 2 {
        return Ok(Series::new(s.name().clone(), [0.0f64]));
    }

    // Collect all pairwise slopes using original indices for time gap
    let mut slopes: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for a in 0..n {
        for b in (a + 1)..n {
            let dx = (indexed_vals[b].0 - indexed_vals[a].0) as f64;
            slopes.push((indexed_vals[b].1 - indexed_vals[a].1) / dx);
        }
    }

    // Sort using total ordering (NaNs are already excluded)
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = slopes.len();
    let median = if len.is_multiple_of(2) {
        (slopes[len / 2 - 1] + slopes[len / 2]) / 2.0
    } else {
        slopes[len / 2]
    };

    Ok(Series::new(s.name().clone(), [median]))
}
