"""
Tests for utils module functions
"""

import pytest
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
import os
import tempfile

from pandoravisibility.utils import (
    analyze_yearly_visibility,
    find_continuous_periods,
    calculate_visibility_statistics,
    find_optimal_observation_windows,
    export_visibility_periods,
    analyze_target_yearly_visibility,
    get_visibility_summary,
)


class TestAnalyzeYearlyVisibility:
    """Test suite for analyze_yearly_visibility function."""

    @pytest.fixture
    def tle_data(self):
        """Standard TLE data for testing."""
        line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
        line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"
        return line1, line2

    @pytest.fixture
    def target_coord(self):
        """Standard target coordinate."""
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    def test_basic_analysis(self, tle_data, target_coord):
        """Test basic yearly visibility analysis."""
        line1, line2 = tle_data
        results = analyze_yearly_visibility(
            line1,
            line2,
            target_coord,
            start_time=Time("2025-01-01T00:00:00"),
            duration_days=7,  # Short duration for faster testing
            time_resolution_hours=2,
            visibility_threshold_hours=4,
            verbose=False,
        )

        assert isinstance(results, dict)
        assert "times" in results
        assert "visibility" in results
        assert "continuous_periods" in results
        assert "statistics" in results
        assert "target_coord" in results
        assert "analysis_params" in results

    def test_returns_correct_time_array(self, tle_data, target_coord):
        """Test that time array has correct length."""
        line1, line2 = tle_data
        duration_days = 7
        time_resolution_hours = 2

        results = analyze_yearly_visibility(
            line1,
            line2,
            target_coord,
            start_time=Time("2025-01-01T00:00:00"),
            duration_days=duration_days,
            time_resolution_hours=time_resolution_hours,
            visibility_threshold_hours=4,
            verbose=False,
        )

        expected_points = int((duration_days * 24) / time_resolution_hours)
        assert len(results["times"]) == expected_points

    def test_visibility_is_boolean_array(self, tle_data, target_coord):
        """Test that visibility results are boolean."""
        line1, line2 = tle_data
        results = analyze_yearly_visibility(
            line1,
            line2,
            target_coord,
            start_time=Time("2025-01-01T00:00:00"),
            duration_days=2,
            time_resolution_hours=2,
            verbose=False,
        )

        assert isinstance(results["visibility"], np.ndarray)
        assert (
            results["visibility"].dtype == bool
            or results["visibility"].dtype == np.bool_
        )

    def test_default_start_time(self, tle_data, target_coord):
        """Test that default start time is Time.now()."""
        line1, line2 = tle_data
        results = analyze_yearly_visibility(
            line1,
            line2,
            target_coord,
            duration_days=1,
            time_resolution_hours=6,
            verbose=False,
        )

        # Should not raise error and should have results
        assert "times" in results
        assert len(results["times"]) > 0

    def test_string_start_time_conversion(self, tle_data, target_coord):
        """Test that string start time is converted to Time object."""
        line1, line2 = tle_data
        results = analyze_yearly_visibility(
            line1,
            line2,
            target_coord,
            start_time="2025-06-01T12:00:00",
            duration_days=1,
            time_resolution_hours=6,
            verbose=False,
        )

        assert isinstance(results["analysis_params"]["start_time"], Time)

    def test_statistics_structure(self, tle_data, target_coord):
        """Test that statistics dict has expected keys."""
        line1, line2 = tle_data
        results = analyze_yearly_visibility(
            line1,
            line2,
            target_coord,
            start_time=Time("2025-01-01T00:00:00"),
            duration_days=2,
            time_resolution_hours=1,
            visibility_threshold_hours=2,
            verbose=False,
        )

        stats = results["statistics"]
        expected_keys = [
            "total_time_hours",
            "total_time_days",
            "total_visible_hours",
            "visibility_percentage",
            "continuous_periods_count",
            "longest_continuous_period_hours",
            "average_continuous_period_hours",
        ]

        for key in expected_keys:
            assert key in stats

    def test_verbose_mode(self, tle_data, target_coord, capsys):
        """Test that verbose mode prints output."""
        line1, line2 = tle_data
        analyze_yearly_visibility(
            line1,
            line2,
            target_coord,
            start_time=Time("2025-01-01T00:00:00"),
            duration_days=1,
            time_resolution_hours=6,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Analyzing yearly visibility" in captured.out


class TestFindContinuousPeriods:
    """Test suite for find_continuous_periods function."""

    def test_single_continuous_period(self):
        """Test finding a single continuous visibility period."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour
        visibility = np.array(
            [False, False, True, True, True, True, True, False, False, False]
        )

        periods = find_continuous_periods(times, visibility, min_duration_hours=2)

        assert len(periods) >= 1
        assert all("start_time" in p for p in periods)
        assert all("end_time" in p for p in periods)
        assert all("duration_hours" in p for p in periods)

    def test_multiple_continuous_periods(self):
        """Test finding multiple continuous visibility periods."""
        times = Time("2025-01-01T00:00:00") + np.arange(20) * u.hour
        visibility = np.array(
            [
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            ]
        )

        periods = find_continuous_periods(times, visibility, min_duration_hours=2)

        assert len(periods) >= 2  # Should find at least 2 periods

    def test_no_periods_found(self):
        """Test when no continuous periods meet minimum duration."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour
        visibility = np.array(
            [False, True, False, True, False, True, False, True, False, False]
        )

        periods = find_continuous_periods(times, visibility, min_duration_hours=5)

        assert len(periods) == 0

    def test_all_visible(self):
        """Test when entire period is visible."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour
        visibility = np.array([True] * 10)

        periods = find_continuous_periods(times, visibility, min_duration_hours=5)

        assert len(periods) >= 1
        first_period = periods[0]
        assert first_period["duration_hours"] >= 5

    def test_period_indices(self):
        """Test that period indices are correctly stored."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour
        visibility = np.array(
            [False, False, True, True, True, True, False, False, False, False]
        )

        periods = find_continuous_periods(times, visibility, min_duration_hours=1)

        assert len(periods) >= 1
        first_period = periods[0]
        assert "start_index" in first_period
        assert "end_index" in first_period
        assert first_period["start_index"] < first_period["end_index"]


class TestCalculateVisibilityStatistics:
    """Test suite for calculate_visibility_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        times = Time("2025-01-01T00:00:00") + np.arange(24) * u.hour
        visibility = np.array([True] * 12 + [False] * 12)
        continuous_periods = []

        stats = calculate_visibility_statistics(
            times, visibility, continuous_periods, min_duration_hours=2
        )

        assert "total_time_hours" in stats
        assert "total_visible_hours" in stats
        assert "visibility_percentage" in stats
        assert stats["visibility_percentage"] > 0

    def test_visibility_percentage_calculation(self):
        """Test that visibility percentage is calculated correctly."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour
        visibility = np.array([True] * 5 + [False] * 5)  # 50% visible
        continuous_periods = []

        stats = calculate_visibility_statistics(
            times, visibility, continuous_periods, min_duration_hours=1
        )

        assert 45 <= stats["visibility_percentage"] <= 55  # Should be around 50%

    def test_with_continuous_periods(self):
        """Test statistics with continuous periods."""
        times = Time("2025-01-01T00:00:00") + np.arange(24) * u.hour
        visibility = np.array([True] * 24)
        continuous_periods = [
            {"duration_hours": 10, "duration_days": 10 / 24},
            {"duration_hours": 8, "duration_days": 8 / 24},
        ]

        stats = calculate_visibility_statistics(
            times, visibility, continuous_periods, min_duration_hours=2
        )

        assert stats["continuous_periods_count"] == 2
        assert stats["longest_continuous_period_hours"] == 10
        assert stats["average_continuous_period_hours"] == 9.0

    def test_no_continuous_periods(self):
        """Test statistics when no continuous periods exist."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour
        visibility = np.array([True, False] * 5)
        continuous_periods = []

        stats = calculate_visibility_statistics(
            times, visibility, continuous_periods, min_duration_hours=5
        )

        assert stats["continuous_periods_count"] == 0
        assert stats["longest_continuous_period_hours"] == 0
        assert stats["average_continuous_period_hours"] == 0


class TestFindOptimalObservationWindows:
    """Test suite for find_optimal_observation_windows function."""

    def test_basic_window_finding(self):
        """Test finding basic observation windows."""
        times = Time("2025-01-01T00:00:00") + np.arange(100) * u.hour
        visibility = np.array([True] * 100)

        continuous_periods = [
            {
                "start_time": times[0],
                "end_time": times[99],
                "duration_hours": 99.0,
            }
        ]

        results = {
            "times": times,
            "visibility": visibility,
            "continuous_periods": continuous_periods,
        }

        windows = find_optimal_observation_windows(
            results, observation_duration_hours=6, min_gap_hours=2
        )

        assert len(windows) > 0
        assert all("start_time" in w for w in windows)
        assert all("end_time" in w for w in windows)
        assert all("duration_hours" in w for w in windows)

    def test_no_windows_short_period(self):
        """Test when period is too short for observation."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour
        visibility = np.array([True] * 10)

        continuous_periods = [
            {
                "start_time": times[0],
                "end_time": times[9],
                "duration_hours": 9.0,
            }
        ]

        results = {
            "times": times,
            "visibility": visibility,
            "continuous_periods": continuous_periods,
        }

        windows = find_optimal_observation_windows(
            results, observation_duration_hours=20, min_gap_hours=2
        )

        assert len(windows) == 0

    def test_window_duration_matches_request(self):
        """Test that observation windows have requested duration."""
        times = Time("2025-01-01T00:00:00") + np.arange(100) * u.hour
        visibility = np.array([True] * 100)

        continuous_periods = [
            {
                "start_time": times[0],
                "end_time": times[99],
                "duration_hours": 99.0,
            }
        ]

        results = {
            "times": times,
            "visibility": visibility,
            "continuous_periods": continuous_periods,
        }

        requested_duration = 8
        windows = find_optimal_observation_windows(
            results, observation_duration_hours=requested_duration, min_gap_hours=2
        )

        assert all(w["duration_hours"] == requested_duration for w in windows)

    def test_multiple_periods(self):
        """Test finding windows across multiple continuous periods."""
        times = Time("2025-01-01T00:00:00") + np.arange(100) * u.hour
        visibility = np.array([True] * 100)

        continuous_periods = [
            {
                "start_time": times[0],
                "end_time": times[40],
                "duration_hours": 40.0,
            },
            {
                "start_time": times[60],
                "end_time": times[99],
                "duration_hours": 39.0,
            },
        ]

        results = {
            "times": times,
            "visibility": visibility,
            "continuous_periods": continuous_periods,
        }

        windows = find_optimal_observation_windows(
            results, observation_duration_hours=10, min_gap_hours=2
        )

        # Should find windows in both periods
        assert len(windows) > 0


class TestExportVisibilityPeriods:
    """Test suite for export_visibility_periods function."""

    def test_export_creates_csv(self):
        """Test that export creates a CSV file."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour

        results = {
            "times": times,
            "continuous_periods": [
                {
                    "start_time": times[0],
                    "end_time": times[5],
                    "duration_hours": 5.0,
                    "duration_days": 5.0 / 24,
                },
                {
                    "start_time": times[6],
                    "end_time": times[9],
                    "duration_hours": 3.0,
                    "duration_days": 3.0 / 24,
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            df = export_visibility_periods(results, filename)

            assert os.path.exists(filename)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "start_time_iso" in df.columns
            assert "end_time_iso" in df.columns
            assert "duration_hours" in df.columns
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_export_dataframe_content(self):
        """Test that exported DataFrame has correct content."""
        times = Time("2025-01-01T00:00:00") + np.arange(10) * u.hour

        results = {
            "times": times,
            "continuous_periods": [
                {
                    "start_time": times[0],
                    "end_time": times[5],
                    "duration_hours": 5.0,
                    "duration_days": 5.0 / 24,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            df = export_visibility_periods(results, filename)

            assert df["period_number"].iloc[0] == 1
            assert df["duration_hours"].iloc[0] == 5.0
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_export_empty_periods(self):
        """Test exporting when no periods exist."""
        results = {
            "times": Time("2025-01-01T00:00:00") + np.arange(10) * u.hour,
            "continuous_periods": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            df = export_visibility_periods(results, filename)

            assert os.path.exists(filename)
            assert len(df) == 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestAnalyzeTargetYearlyVisibility:
    """Test suite for analyze_target_yearly_visibility function."""

    @pytest.fixture
    def tle_data(self):
        """Standard TLE data for testing."""
        line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
        line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"
        return line1, line2

    def test_complete_workflow(self, tle_data, capsys):
        """Test complete analysis workflow."""
        line1, line2 = tle_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            os.chdir(tmpdir)

            try:
                results = analyze_target_yearly_visibility(
                    79.17305002,
                    45.99514569,
                    line1,
                    line2,
                    start_time="2025-01-01T00:00:00",
                    duration_days=3,
                    time_resolution_hours=2,
                    visibility_threshold_hours=4,
                    verbose=False,
                )

                assert "analysis" in results
                assert "observation_windows" in results
                assert "periods_dataframe" in results

                # Check that CSV was created
                csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
                assert len(csv_files) > 0
            finally:
                os.chdir(original_dir)

    def test_string_time_conversion(self, tle_data):
        """Test that string start time is converted properly."""
        line1, line2 = tle_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            os.chdir(tmpdir)

            try:
                results = analyze_target_yearly_visibility(
                    79.17305002,
                    45.99514569,
                    line1,
                    line2,
                    start_time="2025-06-15T12:00:00",
                    duration_days=1,
                    time_resolution_hours=6,
                    verbose=False,
                )

                assert isinstance(
                    results["analysis"]["analysis_params"]["start_time"], Time
                )
            finally:
                os.chdir(original_dir)


class TestGetVisibilitySummary:
    """Test suite for get_visibility_summary function."""

    def test_summary_format(self):
        """Test that summary has expected format."""
        times = Time("2025-01-01T00:00:00") + np.arange(24) * u.hour
        visibility = np.array([True] * 12 + [False] * 12)
        target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

        results = {
            "times": times,
            "visibility": visibility,
            "target_coord": target_coord,
            "continuous_periods": [],
            "statistics": {
                "total_time_hours": 23.0,
                "total_time_days": 23.0 / 24,
                "total_visible_hours": 12.0,
                "visibility_percentage": 50.0,
                "continuous_periods_count": 0,
                "longest_continuous_period_hours": 0,
                "longest_continuous_period_days": 0,
                "average_continuous_period_hours": 0,
                "total_continuous_hours": 0,
                "continuous_visibility_percentage": 0,
                "time_resolution_hours": 1,
            },
            "analysis_params": {
                "start_time": times[0],
                "duration_days": 1,
                "time_resolution_hours": 1,
                "visibility_threshold_hours": 2,
            },
        }

        summary = get_visibility_summary(results)

        assert isinstance(summary, str)
        assert "VISIBILITY ANALYSIS SUMMARY" in summary
        assert "Target:" in summary
        assert "Analysis period:" in summary
        assert "Overall Visibility:" in summary

    def test_summary_contains_statistics(self):
        """Test that summary contains key statistics."""
        times = Time("2025-01-01T00:00:00") + np.arange(24) * u.hour
        visibility = np.array([True] * 24)
        target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

        results = {
            "times": times,
            "visibility": visibility,
            "target_coord": target_coord,
            "continuous_periods": [
                {
                    "start_time": times[0],
                    "end_time": times[23],
                    "duration_hours": 23.0,
                    "duration_days": 23.0 / 24,
                }
            ],
            "statistics": {
                "total_time_hours": 23.0,
                "total_time_days": 23.0 / 24,
                "total_visible_hours": 23.0,
                "visibility_percentage": 100.0,
                "continuous_periods_count": 1,
                "longest_continuous_period_hours": 23.0,
                "longest_continuous_period_days": 23.0 / 24,
                "average_continuous_period_hours": 23.0,
                "total_continuous_hours": 23.0,
                "continuous_visibility_percentage": 100.0,
                "time_resolution_hours": 1,
            },
            "analysis_params": {
                "start_time": times[0],
                "duration_days": 1,
                "time_resolution_hours": 1,
                "visibility_threshold_hours": 2,
            },
        }

        summary = get_visibility_summary(results)

        assert "100.0%" in summary
        assert "Continuous Periods" in summary
