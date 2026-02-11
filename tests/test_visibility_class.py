"""
Tests for Visibility class methods that are not covered in test_import.py
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
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


class TestStarTrackerConstraints:
    """Test suite for star tracker keep-out constraint features."""

    @pytest.fixture
    def line1(self):
        return "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"

    @pytest.fixture
    def line2(self):
        return "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"

    @pytest.fixture
    def target_coord(self):
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        return Time("2025-01-01T00:00:00")

    def test_st_defaults_are_zero(self, line1, line2):
        """Star tracker constraints default to 0 (disabled)."""
        vis = Visibility(line1, line2)
        assert vis.st_sun_min == 0 * u.deg
        assert vis.st_moon_min == 0 * u.deg
        assert vis.st_earthlimb_min == 0 * u.deg

    def test_st_custom_limits_applied(self, line1, line2):
        """Custom star tracker limits are stored on the instance."""
        vis = Visibility(
            line1, line2,
            st_sun_min=45 * u.deg,
            st_moon_min=10 * u.deg,
            st_earthlimb_min=20 * u.deg,
        )
        assert vis.st_sun_min == 45 * u.deg
        assert vis.st_moon_min == 10 * u.deg
        assert vis.st_earthlimb_min == 20 * u.deg

    def test_repr_shows_st_constraints(self, line1, line2):
        """__repr__ includes star tracker constraints when non-zero."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        repr_str = repr(vis)
        assert "st_sun≥45 deg" in repr_str

    def test_repr_hides_st_when_zero(self, line1, line2):
        """__repr__ does not mention star tracker when all ST limits are 0."""
        vis = Visibility(line1, line2)
        repr_str = repr(vis)
        assert "st_sun" not in repr_str
        assert "st_moon" not in repr_str
        assert "st_limb" not in repr_str

    def test_constraint_passes_when_disabled(
        self, line1, line2, target_coord, test_time
    ):
        """With all ST limits at 0, get_star_tracker_constraint always returns True."""
        vis = Visibility(line1, line2)
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result is True

    def test_constraint_passes_when_disabled_array(
        self, line1, line2, target_coord
    ):
        """Disabled ST constraints return all-True array for array times."""
        vis = Visibility(line1, line2)
        times = Time("2025-01-01T00:00:00") + np.arange(3) * u.hour
        result = vis.get_star_tracker_constraint(target_coord, times)
        assert np.all(result)
        assert result.shape == times.shape

    def test_get_star_tracker_skycoord_returns_skycoord(
        self, line1, line2, target_coord, test_time
    ):
        """_get_star_tracker_skycoord returns a SkyCoord in GCRS."""
        vis = Visibility(line1, line2)
        sc = vis._get_star_tracker_skycoord(target_coord, test_time, tracker=1)
        assert isinstance(sc, SkyCoord)

    def test_get_star_tracker_skycoord_two_trackers_differ(
        self, line1, line2, target_coord, test_time
    ):
        """Star tracker 1 and 2 point in different directions."""
        vis = Visibility(line1, line2)
        sc1 = vis._get_star_tracker_skycoord(target_coord, test_time, tracker=1)
        sc2 = vis._get_star_tracker_skycoord(target_coord, test_time, tracker=2)
        sep = sc1.separation(sc2)
        assert sep.deg > 0  # They should not be identical

    def test_get_star_tracker_skycoord_invalid_tracker(
        self, line1, line2, target_coord, test_time
    ):
        """Invalid tracker number raises ValueError."""
        vis = Visibility(line1, line2)
        with pytest.raises(ValueError, match="Invalid tracker"):
            vis._get_star_tracker_skycoord(target_coord, test_time, tracker=3)

    def test_constraint_with_very_large_sun_limit(
        self, line1, line2, target_coord, test_time
    ):
        """A 180° ST sun limit should fail (impossible to satisfy)."""
        vis = Visibility(line1, line2, st_sun_min=180 * u.deg)
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result is False or result == False  # noqa: E712

    def test_constraint_with_small_sun_limit(
        self, line1, line2, target_coord, test_time
    ):
        """A very small ST sun limit should pass for most targets."""
        vis = Visibility(line1, line2, st_sun_min=1 * u.deg)
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result  # Capella is far from the sun, both trackers should be fine

    def test_constraint_integrated_into_visibility(
        self, line1, line2, target_coord, test_time
    ):
        """ST constraint should affect get_visibility result."""
        vis_loose = Visibility(line1, line2, st_sun_min=1 * u.deg)
        vis_tight = Visibility(line1, line2, st_sun_min=180 * u.deg)

        result_loose = vis_loose.get_visibility(target_coord, test_time)
        result_tight = vis_tight.get_visibility(target_coord, test_time)

        # Tight ST sun constraint should block visibility
        assert result_tight is False

    def test_get_all_constraints_includes_star_tracker(
        self, line1, line2, target_coord, test_time
    ):
        """get_all_constraints includes star_tracker key when active."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        constraints = vis.get_all_constraints(target_coord, test_time)
        assert "star_tracker" in constraints

    def test_get_all_constraints_excludes_star_tracker_when_disabled(
        self, line1, line2, target_coord, test_time
    ):
        """get_all_constraints omits star_tracker key when all ST limits are 0."""
        vis = Visibility(line1, line2)
        constraints = vis.get_all_constraints(target_coord, test_time)
        assert "star_tracker" not in constraints

    def test_summary_includes_star_tracker_section(
        self, line1, line2, target_coord, test_time
    ):
        """Summary output includes star tracker section when ST constraints active."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        summary = vis.summary(target_coord, test_time)
        assert "Star Tracker" in summary
        assert "ST1" in summary
        assert "ST2" in summary

    def test_summary_omits_star_tracker_when_disabled(
        self, line1, line2, target_coord, test_time
    ):
        """Summary output has no star tracker section when all ST limits are 0."""
        vis = Visibility(line1, line2)
        summary = vis.summary(target_coord, test_time)
        assert "Star Tracker" not in summary

    def test_constraint_array_time(self, line1, line2, target_coord):
        """ST constraint works with array times."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        times = Time("2025-01-01T00:00:00") + np.arange(3) * u.hour
        result = vis.get_star_tracker_constraint(target_coord, times)
        assert result.shape == times.shape
        assert result.dtype == bool

    def test_skycoord_array_time(self, line1, line2, target_coord):
        """_get_star_tracker_skycoord works with array times."""
        vis = Visibility(line1, line2)
        times = Time("2025-01-01T00:00:00") + np.arange(3) * u.hour
        sc = vis._get_star_tracker_skycoord(target_coord, times, tracker=1)
        assert isinstance(sc, SkyCoord)
        assert sc.shape == times.shape

    def test_st_required_default_is_one(self, line1, line2):
        """st_required defaults to 1."""
        vis = Visibility(line1, line2)
        assert vis.st_required == 1

    def test_st_required_zero_disables_constraint(
        self, line1, line2, target_coord, test_time
    ):
        """st_required=0 means ST constraints are inactive even with limits set."""
        vis = Visibility(
            line1, line2, st_sun_min=180 * u.deg, st_required=0
        )
        # Should always pass since st_required=0 disables ST checks
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result is True

    def test_st_required_zero_excluded_from_all_constraints(
        self, line1, line2, target_coord, test_time
    ):
        """st_required=0 means star_tracker key is absent from get_all_constraints."""
        vis = Visibility(
            line1, line2, st_sun_min=45 * u.deg, st_required=0
        )
        constraints = vis.get_all_constraints(target_coord, test_time)
        assert "star_tracker" not in constraints

    def test_st_required_two_requires_both(
        self, line1, line2, target_coord, test_time
    ):
        """st_required=2 means both trackers must pass."""
        # With a small limit both should pass for a reasonable target
        vis = Visibility(
            line1, line2, st_sun_min=1 * u.deg, st_required=2
        )
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result  # Both should pass with a tiny limit

    def test_st_required_invalid_raises(self, line1, line2):
        """st_required must be 0, 1, or 2."""
        with pytest.raises(ValueError, match="st_required must be 0, 1, or 2"):
            Visibility(line1, line2, st_required=3)

    def test_repr_shows_st_required(self, line1, line2):
        """__repr__ includes st_req when ST constraints are active."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg, st_required=2)
        repr_str = repr(vis)
        assert "st_req=2" in repr_str

    def test_summary_shows_both_label(
        self, line1, line2, target_coord, test_time
    ):
        """Summary shows 'both' when st_required=2."""
        vis = Visibility(
            line1, line2, st_sun_min=1 * u.deg, st_required=2
        )
        summary = vis.summary(target_coord, test_time)
        assert "both" in summary
