use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::PyResult;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

/// Compute the normalized cross-correlation (NCC) between two series using FFT.
///
/// NCC(s) = CC(s) / sqrt(sum(a^2) * sum(b^2))
///
/// Returns the maximum NCC value across all shifts.
/// SBD distance = 1 - max(NCC), so 0 = identical shape, 2 = opposite shape.
fn sbd_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return 0.0;
    }

    // Norm terms
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let denom = norm_a * norm_b;

    // FFT-based cross-correlation
    // Pad to length >= n + m - 1, rounded up to next power of 2 for FFT efficiency
    let cc_len = n + m - 1;
    let fft_len = cc_len.next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    // Zero-pad a and b into complex buffers
    let mut fa: Vec<Complex<f64>> = a.iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)).take(fft_len - n))
        .collect();

    let mut fb: Vec<Complex<f64>> = b.iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)).take(fft_len - m))
        .collect();

    // Forward FFT
    fft.process(&mut fa);
    fft.process(&mut fb);

    // Multiply FA * conj(FB) element-wise
    let mut fc: Vec<Complex<f64>> = fa.iter()
        .zip(fb.iter())
        .map(|(a, b)| a * b.conj())
        .collect();

    // Inverse FFT
    ifft.process(&mut fc);

    // Normalize by fft_len (rustfft doesn't normalize)
    let scale = 1.0 / fft_len as f64;

    // Find max NCC across all valid shifts
    // Cross-correlation has cc_len values. The shifts correspond to:
    //   indices 0..m-1      → b shifted right (negative lag)
    //   indices fft_len-n+1..fft_len-1 → b shifted left (positive lag)
    // We check all cc_len valid positions.
    let mut max_ncc = f64::NEG_INFINITY;

    // Positive lags: indices 0..m (b shifted right relative to a)
    for i in 0..m {
        let ncc = fc[i].re * scale / denom;
        if ncc > max_ncc {
            max_ncc = ncc;
        }
    }

    // Negative lags: indices fft_len-n+1..fft_len (a shifted right relative to b)
    for i in (fft_len - n + 1)..fft_len {
        let ncc = fc[i].re * scale / denom;
        if ncc > max_ncc {
            max_ncc = ncc;
        }
    }

    // SBD = 1 - max(NCC), clamped to [0, 2]
    (1.0 - max_ncc).clamp(0.0, 2.0)
}

/// Groups a DataFrame by "unique_id" and aggregates the "y" column.
fn get_groups(df: &DataFrame) -> Result<LazyFrame, PolarsError> {
    Ok(df.clone().lazy()
        .select([
            col("unique_id").cast(DataType::String),
            col("y").cast(DataType::Float64)
        ])
        .group_by([col("unique_id")])
        .agg([col("y")])
    )
}

/// Optimized conversion of a grouped DataFrame into a HashMap mapping id -> Vec<f64>.
fn df_to_hashmap(df: &DataFrame) -> HashMap<String, Vec<f64>> {
    let unique_id_col = df.column("unique_id").expect("expected column unique_id");
    let y_col = df.column("y").expect("expected column y");

    let unique_ids: Vec<String> = unique_id_col
        .str()
        .expect("expected utf8 column for unique_id")
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    let y_lists: Vec<Vec<f64>> = y_col
        .list()
        .expect("expected a List type for y")
        .into_iter()
        .map(|opt_series| {
            let series = opt_series.expect("null entry in 'y' list column");
            series
                .f64()
                .expect("expected a f64 Series inside the list")
                .into_no_null_iter()
                .collect::<Vec<f64>>()
        })
        .collect();

    assert_eq!(unique_ids.len(), y_lists.len(), "Mismatched lengths in unique_ids and y_lists");

    let hashmap: HashMap<String, Vec<f64>> = (0..unique_ids.len())
        .into_par_iter()
        .map(|i| (unique_ids[i].clone(), y_lists[i].clone()))
        .collect();
    hashmap
}

/// Compute pairwise SBD (Shape-Based Distance) between time series in two DataFrames.
///
/// SBD uses FFT-based normalized cross-correlation to measure shape similarity,
/// invariant to scaling and shifting. Distance range is [0, 2] where
/// 0 = identical shape, 2 = perfectly opposite shape.
///
/// # Arguments
/// * `input1` - First PyDataFrame with columns "unique_id" and "y".
/// * `input2` - Second PyDataFrame with columns "unique_id" and "y".
///
/// # Returns
/// A PyDataFrame with columns "id_1", "id_2", and "sbd".
#[pyfunction]
#[pyo3(signature = (input1, input2))]
pub fn compute_pairwise_sbd(input1: PyDataFrame, input2: PyDataFrame) -> PyResult<PyDataFrame> {
    let df_1: DataFrame = input1.into();
    let df_2: DataFrame = input2.into();

    let uid_a_dtype = df_1.column("unique_id")
        .expect("df_a must have unique_id")
        .dtype().clone();

    let uid_b_dtype = df_2.column("unique_id")
        .expect("df_b must have unique_id")
        .dtype().clone();

    let df_a = df_1
        .lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect().unwrap();

    let df_b = df_2
        .lazy()
        .with_column(col("unique_id").cast(DataType::String))
        .collect().unwrap();

    let grouped_a = get_groups(&df_a).unwrap().collect().unwrap();
    let grouped_b = get_groups(&df_b).unwrap().collect().unwrap();

    let raw_map_a = df_to_hashmap(&grouped_a);
    let raw_map_b = df_to_hashmap(&grouped_b);

    let map_a = Arc::new(raw_map_a);
    let map_b = Arc::new(raw_map_b);

    let left_series_by_key: Vec<(&String, &Vec<f64>)> = map_a.iter().collect();
    let right_series_by_key: Vec<(&String, &Vec<f64>)> = map_b.iter().collect();

    let results: Vec<(String, String, f64)> = left_series_by_key
        .par_iter()
        .flat_map(|&(left_key, left_series)| {
            let map_a = Arc::clone(&map_a);
            let map_b = Arc::clone(&map_b);

            right_series_by_key
                .par_iter()
                .filter_map(move |&(right_key, right_series)| {
                    if left_key == right_key {
                        return None;
                    }
                    if map_b.contains_key(left_key) && map_a.contains_key(right_key) {
                        if left_key >= right_key {
                            return None;
                        }
                    }
                    let distance = sbd_distance(left_series, right_series);
                    Some((left_key.clone(), right_key.clone(), distance))
                })
        })
        .collect();

    let id1s: Vec<String> = results.iter().map(|(id1, _, _)| id1.clone()).collect();
    let id2s: Vec<String> = results.iter().map(|(_, id2, _)| id2.clone()).collect();
    let sbd_vals: Vec<f64> = results.iter().map(|(_, _, v)| *v).collect();

    let columns = vec![
        Column::new("id_1".into(), id1s),
        Column::new("id_2".into(), id2s),
        Column::new("sbd".into(), sbd_vals),
    ];
    let out_df = DataFrame::new(columns).unwrap();
    let casted_out_df = out_df.clone().lazy()
        .with_columns(vec![
            col("id_1").cast(uid_a_dtype),
            col("id_2").cast(uid_b_dtype),
        ]).collect().unwrap();
    Ok(PyDataFrame(casted_out_df))
}
