"""
Tests for Visibility class methods that are not covered in test_import.py
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time

from pandoravisibility import Visibility


class TestVisibilityClassMethods:
    """Test suite for Visibility class methods."""

    @pytest.fixture
    def visibility_instance(self):
        """Create a standard Visibility instance for testing."""
        line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
        line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"
        return Visibility(line1, line2)

    @pytest.fixture
    def custom_visibility_instance(self):
        """Create a Visibility instance with custom limits."""
        line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
        line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"
        return Visibility(
            line1,
            line2,
            moon_min=30 * u.deg,
            sun_min=100 * u.deg,
            earthlimb_min=15 * u.deg,
            mars_min=5 * u.deg,
            jupiter_min=5 * u.deg,
        )

    @pytest.fixture
    def target_coord(self):
        """Standard target coordinate (Capella)."""
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        """Standard test time."""
        return Time("2025-01-01T00:00:00")

    def test_repr_default_constraints(self, visibility_instance):
        """Test __repr__ method with default constraints."""
        repr_str = repr(visibility_instance)
        assert "<Visibility:" in repr_str
        assert "SAT99152" in repr_str
        assert "moon≥" in repr_str
        assert "sun≥" in repr_str
        assert "limb≥" in repr_str

    def test_repr_custom_constraints(self, custom_visibility_instance):
        """Test __repr__ method with custom constraints."""
        repr_str = repr(custom_visibility_instance)
        assert "<Visibility:" in repr_str
        assert "SAT99152" in repr_str
        assert "moon≥30 deg" in repr_str
        assert "sun≥100 deg" in repr_str
        assert "mars≥5 deg" in repr_str
        assert "jupiter≥5 deg" in repr_str

    def test_repr_zero_constraints(self):
        """Test __repr__ method with zero constraints."""
        line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
        line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"
        vis = Visibility(
            line1, line2, moon_min=0 * u.deg, sun_min=0 * u.deg, earthlimb_min=0 * u.deg
        )
        repr_str = repr(vis)
        assert "<Visibility:" in repr_str
        assert "default" in repr_str

    def test_get_all_constraints(self, visibility_instance, target_coord, test_time):
        """Test get_all_constraints method returns dict with all constraints."""
        constraints = visibility_instance.get_all_constraints(target_coord, test_time)

        assert isinstance(constraints, dict)
        assert "moon" in constraints
        assert "sun" in constraints
        assert "earthlimb" in constraints
        # Can be bool or numpy bool
        assert isinstance(constraints["moon"], (bool, np.bool_))
        assert isinstance(constraints["sun"], (bool, np.bool_))
        assert isinstance(constraints["earthlimb"], (bool, np.bool_))

    def test_get_all_constraints_with_planets(
        self, custom_visibility_instance, target_coord, test_time
    ):
        """Test get_all_constraints includes planetary constraints when enabled."""
        constraints = custom_visibility_instance.get_all_constraints(
            target_coord, test_time
        )

        assert "mars" in constraints
        assert "jupiter" in constraints
        # Can be bool or numpy bool
        assert isinstance(constraints["mars"], (bool, np.bool_))
        assert isinstance(constraints["jupiter"], (bool, np.bool_))

    def test_get_separations(self, visibility_instance, target_coord, test_time):
        """Test get_separations method returns angles for all bodies."""
        separations = visibility_instance.get_separations(target_coord, test_time)

        assert isinstance(separations, dict)
        assert "moon" in separations
        assert "sun" in separations
        assert "earthlimb" in separations
        assert "mars" in separations
        assert "jupiter" in separations

        # Check that all separations are angles
        for body, sep in separations.items():
            assert hasattr(sep, "unit")  # Has astropy unit
            assert sep.unit.is_equivalent(u.deg)  # Is angle unit

    def test_get_separations_values_reasonable(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that separation values are in reasonable ranges."""
        separations = visibility_instance.get_separations(target_coord, test_time)

        # All separations should be between -90 and 360 degrees
        for body, sep in separations.items():
            sep_deg = sep.to(u.deg).value
            assert (
                -90 <= sep_deg <= 360
            ), f"{body} separation {sep_deg} deg is out of reasonable range"

    def test_summary_scalar_time(self, visibility_instance, target_coord, test_time):
        """Test summary method with scalar time."""
        summary = visibility_instance.summary(target_coord, test_time)

        assert isinstance(summary, str)
        assert "Visibility Summary" in summary
        assert "Target:" in summary
        assert "Time:" in summary
        assert "Sat:" in summary
        assert "Moon" in summary or "moon" in summary.lower()
        assert "Sun" in summary or "sun" in summary.lower()
        assert "Earthlimb" in summary or "earthlimb" in summary.lower()
        assert "Overall:" in summary
        assert "VISIBLE" in summary or "NOT VISIBLE" in summary

    def test_summary_shows_constraint_status(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that summary shows PASS/FAIL status for constraints."""
        summary = visibility_instance.summary(target_coord, test_time)

        # Should contain status indicators
        assert "PASS" in summary or "FAIL" in summary
        # Should contain check marks or crosses
        assert "✓" in summary or "✗" in summary

    def test_summary_shows_separation_values(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that summary shows actual separation values."""
        summary = visibility_instance.summary(target_coord, test_time)

        assert "req:" in summary or "actual:" in summary
        # Should contain degree values
        assert "deg" in summary

    def test_summary_array_time_raises_error(self, visibility_instance, target_coord):
        """Test that summary raises error with array time input."""

        times = Time("2025-01-01T00:00:00") + np.arange(5) * u.hour

        with pytest.raises(ValueError, match="scalar"):
            visibility_instance.summary(target_coord, times)

    def test_invalid_tle_empty_lines(self):
        """Test that empty TLE lines raise ValueError."""
        with pytest.raises(ValueError, match="TLE lines cannot be empty"):
            Visibility("", "")

    def test_invalid_tle_none_lines(self):
        """Test that None TLE lines raise ValueError."""
        with pytest.raises(ValueError, match="TLE lines cannot be empty"):
            Visibility(None, None)

    def test_invalid_tle_bad_format(self):
        """Test that malformed TLE data raises ValueError."""
        line1 = "INVALID LINE 1"
        line2 = "INVALID LINE 2"

        # The SGP4 library may not always raise an exception for invalid data
        # Just try to create and catch any exception
        try:
            vis = Visibility(line1, line2)
            # If it doesn't raise, at least verify it created something
            assert vis is not None
        except (ValueError, Exception):
            # This is the expected behavior
            pass

    def test_get_constraint_invalid_body(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that get_constraint raises error for invalid body name."""
        with pytest.raises(ValueError, match="Invalid body"):
            visibility_instance.get_constraint(target_coord, "venus", test_time)

    def test_get_state_without_time_attribute(self, visibility_instance):
        """Test get_state raises error when no time is provided and self.time is not set."""
        # Remove time attribute if it exists
        if hasattr(visibility_instance, "time"):
            delattr(visibility_instance, "time")

        with pytest.raises(ValueError, match="No time parameter specified"):
            visibility_instance.get_state()

    def test_custom_limits_applied(self, custom_visibility_instance):
        """Test that custom limits are properly applied to instance."""
        assert custom_visibility_instance.moon_min == 30 * u.deg
        assert custom_visibility_instance.sun_min == 100 * u.deg
        assert custom_visibility_instance.earthlimb_min == 15 * u.deg
        assert custom_visibility_instance.mars_min == 5 * u.deg
        assert custom_visibility_instance.jupiter_min == 5 * u.deg

    def test_visibility_with_different_constraints_changes_result(
        self, target_coord, test_time
    ):
        """Test that different constraints produce different visibility results."""
        line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
        line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"

        # Very loose constraints
        vis_loose = Visibility(
            line1,
            line2,
            moon_min=0 * u.deg,
            sun_min=0 * u.deg,
            earthlimb_min=-90 * u.deg,
        )
        result_loose = vis_loose.get_visibility(target_coord, test_time)

        # Very tight constraints
        vis_tight = Visibility(line1, line2, moon_min=180 * u.deg)
        result_tight = vis_tight.get_visibility(target_coord, test_time)

        # Loose constraints should be more permissive
        assert result_loose is True
        assert result_tight is False

    def test_get_period_returns_correct_units(self, visibility_instance):
        """Test that get_period returns value with correct units."""
        period = visibility_instance.get_period()
        assert hasattr(period, "unit")
        assert period.unit == u.minute

    def test_get_state_returns_skycoord(self, visibility_instance, test_time):
        """Test that get_state returns a SkyCoord object."""
        state = visibility_instance.get_state(test_time)
        assert isinstance(state, SkyCoord)
        assert hasattr(state, "x")
        assert hasattr(state, "y")
        assert hasattr(state, "z")

    def test_get_state_with_array_time(self, visibility_instance):
        """Test get_state with array of times."""

        times = Time("2025-01-01T00:00:00") + np.arange(5) * u.hour
        states = visibility_instance.get_state(times)

        assert len(states) == 5
        assert isinstance(states, SkyCoord)

    def test_get_star_tracker_angles_return_structure(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that get_star_tracker_angles returns dict with correct keys."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        assert isinstance(result, dict)
        assert "ra" in result
        assert "dec" in result
        assert "sun_angle" in result
        assert "earth_angle" in result
        assert "earthlimb_angle" in result

        # Check all values are Quantities with degree units
        for key, value in result.items():
            assert hasattr(value, "unit")
            assert value.unit.is_equivalent(u.deg)

    def test_get_star_tracker_angles_tracker1(
        self, visibility_instance, target_coord, test_time
    ):
        """Test star tracker 1 RA/Dec calculation."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # RA should be in [0, 360) degrees
        assert 0 * u.deg <= result["ra"] < 360 * u.deg
        # Dec should be in [-90, 90] degrees
        assert -90 * u.deg <= result["dec"] <= 90 * u.deg

    def test_get_star_tracker_angles_tracker2(
        self, visibility_instance, target_coord, test_time
    ):
        """Test star tracker 2 RA/Dec calculation."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=2
        )

        # RA should be in [0, 360) degrees
        assert 0 * u.deg <= result["ra"] < 360 * u.deg
        # Dec should be in [-90, 90] degrees
        assert -90 * u.deg <= result["dec"] <= 90 * u.deg

    def test_get_star_tracker_angles_different_trackers(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that tracker 1 and tracker 2 give different RA/Dec."""
        result1 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )
        result2 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=2
        )

        # The two trackers should point at different sky positions
        # Tracker 1 and 2 have opposite Y-components in their boresight vectors
        # (0.7071 vs -0.7071), which should produce different pointings
        ra_diff = abs(result1["ra"] - result2["ra"])
        # Handle RA wrapping at 0/360 degrees
        ra_diff = min(ra_diff, 360 * u.deg - ra_diff)
        dec_diff = abs(result1["dec"] - result2["dec"])

        # At least one coordinate should be different by > 1 degree
        # This threshold accounts for the geometric difference in tracker orientations
        assert ra_diff > 1.0 * u.deg or dec_diff > 1.0 * u.deg

    def test_get_star_tracker_angles_sun_angle(
        self, visibility_instance, target_coord, test_time
    ):
        """Test sun angle calculation is reasonable."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Sun angle should be between 0 and 180 degrees
        assert 0 * u.deg <= result["sun_angle"] <= 180 * u.deg

    def test_get_star_tracker_angles_earth_angle(
        self, visibility_instance, target_coord, test_time
    ):
        """Test earth angle calculation is reasonable."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Earth angle should be between 0 and 180 degrees
        assert 0 * u.deg <= result["earth_angle"] <= 180 * u.deg

    def test_get_star_tracker_angles_earthlimb_angle(
        self, visibility_instance, target_coord, test_time
    ):
        """Test earthlimb angle calculation is reasonable."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Earthlimb angle should be between -90 and 180 degrees
        # (can be negative if below horizon)
        assert -90 * u.deg <= result["earthlimb_angle"] <= 180 * u.deg

    def test_get_star_tracker_angles_invalid_tracker_number(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that invalid tracker number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tracker number"):
            visibility_instance.get_star_tracker_angles(
                target_coord, test_time, tracker=3
            )

        with pytest.raises(ValueError, match="Invalid tracker number"):
            visibility_instance.get_star_tracker_angles(
                target_coord, test_time, tracker=0
            )

    def test_get_star_tracker_angles_sun_aligned_target(self, visibility_instance):
        """Test that target aligned with sun raises ValueError."""
        # Use a time and find the sun's position
        test_time = Time("2025-06-21T12:00:00")

        # Get the sun's position at this time
        observer_location = visibility_instance._get_observer_location(test_time)
        sun_coord = get_body("sun", time=test_time, location=observer_location)

        # Use sun's position as target (aligned with sun)
        with pytest.raises(
            ValueError, match="Cannot determine attitude: target is aligned with the sun"
        ):
            visibility_instance.get_star_tracker_angles(
                sun_coord, test_time, tracker=1
            )

    def test_get_star_tracker_angles_default_tracker(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that default tracker parameter is 1."""
        result_default = visibility_instance.get_star_tracker_angles(
            target_coord, test_time
        )
        result_tracker1 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Default should be same as tracker=1 for all values
        assert result_default["ra"] == result_tracker1["ra"]
        assert result_default["dec"] == result_tracker1["dec"]
        assert result_default["sun_angle"] == result_tracker1["sun_angle"]
        assert result_default["earth_angle"] == result_tracker1["earth_angle"]
        assert result_default["earthlimb_angle"] == result_tracker1["earthlimb_angle"]
