"""
Tests for utils module functions
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from pandoravisibility.utils import (
    analyze_yearly_visibility,
    calculate_visibility_statistics,
    find_continuous_periods,
)


class TestAnalyzeYearlyVisibility:
    """Test suite for analyze_yearly_visibility function."""

    @pytest.fixture
    def tle_data(self):
        """Standard TLE data for testing."""
        line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
        line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"
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
