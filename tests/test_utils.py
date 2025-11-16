"""Tests for pandoravisibility utils module."""

import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from pandoravisibility import (
    Visibility,
    analyze_yearly_visibility,
    find_continuous_periods,
    calculate_visibility_statistics,
    find_optimal_observation_windows,
    export_visibility_periods
)


@pytest.fixture
def sample_tle():
    """Provide sample TLE lines for testing."""
    line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
    line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"
    return line1, line2


@pytest.fixture
def sample_target():
    """Provide sample target coordinates (Capella)."""
    return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")


class TestAnalyzeYearlyVisibility:
    """Test analyze_yearly_visibility function."""
    
    def test_basic_analysis(self, sample_tle, sample_target):
        """Test basic yearly visibility analysis."""
        line1, line2 = sample_tle
        
        results = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=7,
            time_resolution_hours=1,
            visibility_threshold_hours=6
        )
        
        assert "times" in results
        assert "visibility" in results
        assert "continuous_periods" in results
        assert "statistics" in results
        assert "target_coord" in results
    
    def test_results_structure(self, sample_tle, sample_target):
        """Test that results have correct structure."""
        line1, line2 = sample_tle
        
        results = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=2,
            time_resolution_hours=2
        )
        
        # Check times array
        assert len(results["times"]) == 24  # 2 days * 24 hours / 2 hour resolution
        
        # Check visibility array
        assert len(results["visibility"]) == len(results["times"])
        assert results["visibility"].dtype == bool
        
        # Check statistics
        stats = results["statistics"]
        assert "total_time_hours" in stats
        assert "visibility_percentage" in stats
        assert "continuous_periods_count" in stats
    
    def test_different_resolutions(self, sample_tle, sample_target):
        """Test with different time resolutions."""
        line1, line2 = sample_tle
        
        # Coarse resolution
        results_coarse = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=1,
            time_resolution_hours=6
        )
        
        # Fine resolution
        results_fine = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=1,
            time_resolution_hours=1
        )
        
        assert len(results_coarse["times"]) == 4
        assert len(results_fine["times"]) == 24
        assert len(results_coarse["times"]) < len(results_fine["times"])


class TestFindContinuousPeriods:
    """Test find_continuous_periods function."""
    
    def test_simple_period(self):
        """Test finding a simple continuous period."""
        times = Time("2025-01-01") + np.arange(10) * u.hour
        visibility = np.array([False, False, True, True, True, True, False, False, False, False])
        
        periods = find_continuous_periods(times, visibility, min_duration_hours=2)
        
        assert len(periods) == 1
        assert periods[0]["duration_hours"] >= 2
        assert periods[0]["start_index"] == 2
        assert periods[0]["end_index"] == 5
    
    def test_multiple_periods(self):
        """Test finding multiple continuous periods."""
        times = Time("2025-01-01") + np.arange(20) * u.hour
        visibility = np.array([
            False, True, True, True, False, False,
            True, True, True, True, True, False,
            False, True, True, False, False, False, False, False
        ])
        
        periods = find_continuous_periods(times, visibility, min_duration_hours=2)
        
        assert len(periods) >= 2
        
        # Check that all periods meet minimum duration
        for period in periods:
            assert period["duration_hours"] >= 2
    
    def test_no_periods(self):
        """Test when no continuous periods exist."""
        times = Time("2025-01-01") + np.arange(10) * u.hour
        visibility = np.zeros(10, dtype=bool)
        
        periods = find_continuous_periods(times, visibility, min_duration_hours=2)
        
        assert len(periods) == 0
    
    def test_entire_period_visible(self):
        """Test when entire period is visible."""
        times = Time("2025-01-01") + np.arange(24) * u.hour
        visibility = np.ones(24, dtype=bool)
        
        periods = find_continuous_periods(times, visibility, min_duration_hours=2)
        
        assert len(periods) == 1
        assert periods[0]["duration_hours"] >= 20  # Should be close to 24


class TestCalculateVisibilityStatistics:
    """Test calculate_visibility_statistics function."""
    
    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        times = Time("2025-01-01") + np.arange(24) * u.hour
        visibility = np.array([True] * 12 + [False] * 12)
        
        period = {
            "start_time": times[0],
            "end_time": times[11],
            "duration_hours": 11.0,
            "duration_days": 11.0/24,
            "start_index": 0,
            "end_index": 11
        }
        
        stats = calculate_visibility_statistics(
            times, visibility, [period], min_duration_hours=2
        )
        
        assert stats["total_time_hours"] == 23.0  # 24 - 1
        assert 45 < stats["visibility_percentage"] < 55  # Should be ~50%
        assert stats["continuous_periods_count"] == 1
        assert stats["longest_continuous_period_hours"] == 11.0
    
    def test_no_visibility(self):
        """Test statistics when nothing is visible."""
        times = Time("2025-01-01") + np.arange(24) * u.hour
        visibility = np.zeros(24, dtype=bool)
        
        stats = calculate_visibility_statistics(
            times, visibility, [], min_duration_hours=2
        )
        
        assert stats["visibility_percentage"] == 0.0
        assert stats["continuous_periods_count"] == 0
        assert stats["longest_continuous_period_hours"] == 0


class TestFindOptimalObservationWindows:
    """Test find_optimal_observation_windows function."""
    
    def test_single_window(self, sample_tle, sample_target):
        """Test finding observation windows."""
        line1, line2 = sample_tle
        
        results = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=2,
            time_resolution_hours=1,
            visibility_threshold_hours=6
        )
        
        windows = find_optimal_observation_windows(
            results,
            observation_duration_hours=3,
            min_gap_hours=1
        )
        
        assert isinstance(windows, list)
        
        # If there are windows, check their structure
        if len(windows) > 0:
            window = windows[0]
            assert "start_time" in window
            assert "end_time" in window
            assert "duration_hours" in window
            assert window["duration_hours"] == 3
    
    def test_no_windows_if_duration_too_long(self, sample_tle, sample_target):
        """Test that no windows are found if observation duration is too long."""
        line1, line2 = sample_tle
        
        results = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=1,
            time_resolution_hours=1,
            visibility_threshold_hours=6
        )
        
        windows = find_optimal_observation_windows(
            results,
            observation_duration_hours=100,  # Unrealistically long
            min_gap_hours=1
        )
        
        # Should have no windows or very few
        assert len(windows) <= len(results["continuous_periods"])


class TestExportVisibilityPeriods:
    """Test export_visibility_periods function."""
    
    def test_export_to_csv(self, sample_tle, sample_target, tmp_path):
        """Test exporting visibility periods to CSV."""
        line1, line2 = sample_tle
        
        results = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=2,
            time_resolution_hours=1,
            visibility_threshold_hours=6
        )
        
        # Export to temporary file
        output_file = tmp_path / "test_visibility.csv"
        df = export_visibility_periods(results, str(output_file))
        
        # Check that file was created
        assert output_file.exists()
        
        # Check DataFrame structure
        assert "period_number" in df.columns
        assert "start_time_iso" in df.columns
        assert "end_time_iso" in df.columns
        assert "duration_hours" in df.columns
        
        # Check that number of rows matches periods
        assert len(df) == len(results["continuous_periods"])


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self, sample_tle, sample_target, tmp_path):
        """Test complete analysis workflow."""
        line1, line2 = sample_tle
        
        # Analyze visibility
        results = analyze_yearly_visibility(
            line1, line2, sample_target,
            start_time="2025-01-01",
            duration_days=7,
            time_resolution_hours=2,
            visibility_threshold_hours=6
        )
        
        # Find observation windows
        windows = find_optimal_observation_windows(
            results,
            observation_duration_hours=4,
            min_gap_hours=2
        )
        
        # Export results
        output_file = tmp_path / "workflow_test.csv"
        df = export_visibility_periods(results, str(output_file))
        
        # Verify everything worked
        assert len(results["times"]) == 84  # 7 days * 24 hours / 2
        assert isinstance(windows, list)
        assert output_file.exists()
        assert len(df) == len(results["continuous_periods"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
