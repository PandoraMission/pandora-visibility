"""Comprehensive tests for pandoravisibility package."""

import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from pandoravisibility import Visibility


# Test fixtures
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


@pytest.fixture
def sample_times():
    """Provide sample time array for testing."""
    tstart = Time("2025-01-01T00:00:00.000")
    tstop = Time("2025-01-02T00:00:00.000")
    dt = TimeDelta((1 / 144) * u.day, format="jd")
    n_steps = int((tstop - tstart) / dt)
    time_deltas = TimeDelta(np.arange(n_steps) * dt.jd, format='jd')
    return tstart + time_deltas


class TestVisibilityInit:
    """Test Visibility class initialization."""
    
    def test_valid_initialization(self, sample_tle):
        """Test initialization with valid TLE."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        assert vis.tle is not None
        assert int(vis.tle.satnum) == 99152
    
    def test_empty_tle_lines(self):
        """Test that empty TLE lines raise ValueError."""
        with pytest.raises(ValueError, match="TLE lines cannot be empty"):
            Visibility("", "")
    
    def test_custom_limits(self, sample_tle):
        """Test initialization with custom constraint limits."""
        line1, line2 = sample_tle
        vis = Visibility(
            line1, line2,
            moon_min=30*u.deg,
            sun_min=85*u.deg,
            earthlimb_min=15*u.deg
        )
        assert vis.moon_min == 30*u.deg
        assert vis.sun_min == 85*u.deg
        assert vis.earthlimb_min == 15*u.deg


class TestVisibilityMethods:
    """Test Visibility class methods."""
    
    def test_get_period(self, sample_tle):
        """Test orbital period calculation."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        period = vis.get_period()
        
        assert isinstance(period, u.Quantity)
        assert period.unit == u.minute
        assert 96.6 < period.value < 96.7
    
    def test_get_state_scalar_time(self, sample_tle):
        """Test state calculation with scalar time."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        time = Time("2025-01-01T00:00:00")
        
        state = vis.get_state(time)
        assert state is not None
        assert hasattr(state, 'x')
    
    def test_repr(self, sample_tle):
        """Test string representation."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        repr_str = repr(vis)
        
        assert "Visibility" in repr_str
        assert "SAT99152" in repr_str


class TestConstraintMethods:
    """Test constraint-related methods."""
    
    def test_get_constraint_moon(self, sample_tle, sample_target):
        """Test moon constraint calculation."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        time = Time("2025-01-01T00:00:00")
        
        result = vis.get_constraint(sample_target, "moon", time)
        assert isinstance(result, (bool, np.bool_))
    
    def test_get_all_constraints(self, sample_tle, sample_target):
        """Test getting all constraints at once."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        time = Time("2025-01-01T00:00:00")
        
        constraints = vis.get_all_constraints(sample_target, time)
        
        assert isinstance(constraints, dict)
        assert "moon" in constraints
        assert "sun" in constraints
        assert "earthlimb" in constraints


class TestVisibilityCalculation:
    """Test visibility calculation methods."""
    
    def test_get_visibility_scalar(self, sample_tle, sample_target):
        """Test visibility calculation with scalar time."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        time = Time("2025-01-01T00:00:00")
        
        result = vis.get_visibility(sample_target, time)
        assert isinstance(result, bool)
    
    def test_get_visibility_array(self, sample_tle, sample_target, sample_times):
        """Test visibility calculation with time array."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        
        result = vis.get_visibility(sample_target, sample_times)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_times)
        assert result.dtype == bool
    
    def test_summary_scalar_time(self, sample_tle, sample_target):
        """Test summary method with scalar time."""
        line1, line2 = sample_tle
        vis = Visibility(line1, line2)
        time = Time("2025-01-01T00:00:00")
        
        summary = vis.summary(sample_target, time)
        
        assert isinstance(summary, str)
        assert "Visibility Summary" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
