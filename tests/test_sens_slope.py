import polars as pl
import pytest

from polars_ts import sens_slope


class TestSensSlopeBasic:
    def test_perfect_linear_upward(self):
        """[0, 1, 2, 3, 4] has slope exactly 1.0."""
        df = pl.DataFrame({"y": [0.0, 1.0, 2.0, 3.0, 4.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(1.0)

    def test_perfect_linear_downward(self):
        """[4, 3, 2, 1, 0] has slope exactly -1.0."""
        df = pl.DataFrame({"y": [4.0, 3.0, 2.0, 1.0, 0.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(-1.0)

    def test_constant_series(self):
        """A constant series has slope 0.0."""
        df = pl.DataFrame({"y": [5.0, 5.0, 5.0, 5.0, 5.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(0.0)

    def test_single_element(self):
        """A single element should return 0.0."""
        df = pl.DataFrame({"y": [42.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(0.0)

    def test_two_elements(self):
        """Two elements: slope = (5 - 1) / (1 - 0) = 4.0."""
        df = pl.DataFrame({"y": [1.0, 5.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(4.0)

    def test_known_slope_with_noise(self):
        """[0, 2, 4, 6, 8] — all pairwise slopes are 2.0, so median is 2.0."""
        df = pl.DataFrame({"y": [0.0, 2.0, 4.0, 6.0, 8.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(2.0)


class TestSensSlopeProperties:
    def test_robust_to_single_outlier(self):
        """An outlier should not heavily affect the median slope."""
        # Without outlier: [0, 1, 2, 3, 4] → slope = 1.0
        df_clean = pl.DataFrame({"y": [0.0, 1.0, 2.0, 3.0, 4.0]})
        # With outlier at index 2
        df_outlier = pl.DataFrame({"y": [0.0, 1.0, 100.0, 3.0, 4.0]})
        clean = df_clean.select(sens_slope("y").alias("ss"))["ss"][0]
        outlier = df_outlier.select(sens_slope("y").alias("ss"))["ss"][0]
        # Median should be much less affected than mean-based slope
        assert abs(outlier - clean) < abs(100.0 - clean)

    def test_negative_slope(self):
        """[10, 7, 4, 1] has slope -3.0."""
        df = pl.DataFrame({"y": [10.0, 7.0, 4.0, 1.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(-3.0)


class TestSensSlopeEdgeCases:
    def test_nulls_preserve_time_gap(self):
        """Nulls should be skipped but original indices preserved for time gap.

        [1.0, null, 3.0] → slope = (3-1)/(2-0) = 1.0 (not 2.0).
        """
        df = pl.DataFrame({"y": [1.0, None, 3.0]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(1.0)

    def test_nan_values_filtered(self):
        """NaN values should be excluded from slope computation."""
        df = pl.DataFrame({"y": [0.0, float("nan"), 2.0, float("nan"), 4.0]})
        result = df.select(sens_slope("y").alias("ss"))
        # Valid pairs: (0,2) at idx 0,2 → slope 1.0; (0,4) at idx 0,4 → slope 1.0;
        # (2,4) at idx 2,4 → slope 1.0. Median = 1.0
        assert result["ss"][0] == pytest.approx(1.0)

    def test_all_nulls_returns_zero(self):
        """All nulls should return 0.0 (fewer than 2 valid values)."""
        df = pl.DataFrame({"y": [None, None, None]}).cast({"y": pl.Float64})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(0.0)

    def test_all_nan_returns_zero(self):
        """All NaN should return 0.0."""
        df = pl.DataFrame({"y": [float("nan"), float("nan")]})
        result = df.select(sens_slope("y").alias("ss"))
        assert result["ss"][0] == pytest.approx(0.0)


class TestSensSlopeWithGroupBy:
    def test_group_by_usage(self):
        """Sen's slope should work inside a group_by context."""
        df = pl.DataFrame(
            {
                "group": ["A"] * 5 + ["B"] * 5,
                "y": [0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 6.0, 4.0, 2.0, 0.0],
            }
        )
        result = df.group_by("group").agg(sens_slope("y").alias("ss")).sort("group").explode("ss")

        ss_a = result.filter(pl.col("group") == "A")["ss"].to_list()[0]
        assert ss_a == pytest.approx(1.0)

        ss_b = result.filter(pl.col("group") == "B")["ss"].to_list()[0]
        assert ss_b == pytest.approx(-2.0)
