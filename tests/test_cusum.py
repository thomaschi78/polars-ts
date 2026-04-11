import polars as pl
import pytest

from polars_ts import cusum


class TestCUSUMBasic:
    def test_constant_series_zero(self):
        """A constant series has zero deviation from mean → CUSUM is all zeros."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "y": [3.0, 3.0, 3.0, 3.0, 3.0],
            }
        )
        result = cusum(df)
        assert "cusum" in result.columns
        assert result["cusum"].to_list() == pytest.approx([0.0] * 5)

    def test_step_change_detected(self):
        """A step change should produce a clear inflection in CUSUM."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 6,
                "y": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
            }
        )
        result = cusum(df, normalize=False)
        cusum_vals = result["cusum"].to_list()
        # Mean is 3.0. Deviations: [-2, -2, -2, 2, 2, 2]
        # Cumsum: [-2, -4, -6, -4, -2, 0]
        assert cusum_vals == pytest.approx([-2.0, -4.0, -6.0, -4.0, -2.0, 0.0])

    def test_linear_trend(self):
        """A linear trend should produce monotonically changing CUSUM at the tails."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = cusum(df, normalize=False)
        cusum_vals = result["cusum"].to_list()
        # Mean = 3.0. Deviations: [-2, -1, 0, 1, 2]
        # Cumsum: [-2, -3, -3, -2, 0]
        assert cusum_vals == pytest.approx([-2.0, -3.0, -3.0, -2.0, 0.0])

    def test_single_element(self):
        """A single element has zero deviation."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"],
                "y": [42.0],
            }
        )
        result = cusum(df, normalize=False)
        assert result["cusum"][0] == pytest.approx(0.0)

    def test_output_preserves_columns(self):
        """The original columns should be preserved."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": [1, 2, 3],
                "y": [1.0, 2.0, 3.0],
            }
        )
        result = cusum(df)
        assert set(df.columns).issubset(set(result.columns))
        assert "cusum" in result.columns


class TestCUSUMGroupBy:
    def test_independent_groups(self):
        """Each group should have its own CUSUM computed independently."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 1.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        result = cusum(df, normalize=False)
        cusum_a = result.filter(pl.col("unique_id") == "A")["cusum"].to_list()
        cusum_b = result.filter(pl.col("unique_id") == "B")["cusum"].to_list()
        # Group A: mean=3, deviations=[-2,-2,2,2], cumsum=[-2,-4,-2,0]
        assert cusum_a == pytest.approx([-2.0, -4.0, -2.0, 0.0])
        # Group B: constant → all zeros
        assert cusum_b == pytest.approx([0.0, 0.0, 0.0, 0.0])


class TestCUSUMNormalization:
    def test_normalized_vs_unnormalized(self):
        """Normalized CUSUM should be unnormalized divided by std."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4,
                "y": [1.0, 1.0, 5.0, 5.0],
            }
        )
        raw = cusum(df, normalize=False)["cusum"].to_list()
        normed = cusum(df, normalize=True)["cusum"].to_list()
        # They should differ in scale but have the same sign pattern
        for r, n in zip(raw, normed, strict=False):
            if r != 0:
                assert (r > 0) == (n > 0)

    def test_normalized_is_unitless(self):
        """Scaling all values should not change normalized CUSUM."""
        df1 = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 1.0, 5.0, 5.0]})
        df2 = pl.DataFrame({"unique_id": ["A"] * 4, "y": [100.0, 100.0, 500.0, 500.0]})
        c1 = cusum(df1, normalize=True)["cusum"].to_list()
        c2 = cusum(df2, normalize=True)["cusum"].to_list()
        for a, b in zip(c1, c2, strict=False):
            assert a == pytest.approx(b, abs=1e-10)


class TestCUSUMValidation:
    def test_empty_dataframe_raises(self):
        df = pl.DataFrame({"unique_id": [], "y": []}).cast({"unique_id": pl.String, "y": pl.Float64})
        with pytest.raises(ValueError):
            cusum(df)

    def test_missing_column_raises(self):
        df = pl.DataFrame({"unique_id": ["A"], "value": [1.0]})
        with pytest.raises(KeyError):
            cusum(df)
