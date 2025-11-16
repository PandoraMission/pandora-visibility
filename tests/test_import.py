from pandoravisibility import Visibility
import numpy as np
from packaging import version


def test_numpy_compatibility():
    """Test that numpy version is >= 1.26 and imports work correctly."""
    numpy_version = version.parse(np.__version__)
    min_version = version.parse("1.26.0")
    assert numpy_version >= min_version, \
        f"NumPy version {np.__version__} is less than 1.26.0"

    # Verify numpy can be imported and basic operations work
    test_array = np.array([1, 2, 3])
    assert test_array.sum() == 6
    assert test_array.shape == (3,)


def test_visibility():
    # Example TLE lines (replace with actual TLE data)
    line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
    line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"

    vis = Visibility(line1, line2)

    # Test get_period method
    period = vis.get_period()
    assert period.value > 96.68
    assert period.value < 96.69

    # Test get_state method with a time input
    from astropy.time import Time

    time = Time("2025-01-01T00:00:00")
    state = vis.get_state(time)
    assert state is not None

    # test satnum
    assert int(vis.tle.satnum) == 99152

    from astropy.constants import R_earth
    from astropy import units as u

    assert (vis.tle.a * u.earthRad - R_earth).to(u.km).value > 596
    assert (vis.tle.a * u.earthRad - R_earth).to(u.km).value < 597


def test_target():
    # Example TLE lines
    line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
    line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"

    vis = Visibility(line1, line2)

    from astropy.time import Time, TimeDelta

    tstart = Time("2025-01-01T00:00:00.000")
    tstop = Time("2025-01-02T00:00:00.000")  # Example stop time

    import numpy as np
    from astropy import units as u

    dt = TimeDelta((1 / 144) * u.day, format="jd")  # 10 minutes in Julian days
    n_steps = int((tstop - tstart) / dt)  # Number of time steps
    time_deltas = TimeDelta(np.arange(n_steps) * dt.jd, format='jd')
    times = tstart + time_deltas

    # using Capella because it is visible in the time period
    from astropy.coordinates import SkyCoord

    target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")
    targ_vis = vis.get_visibility(target_coord, times)

    assert int(targ_vis.shape[0]) == 144
    assert targ_vis.astype(int).sum() == 75


def test_custom_limits():
    # Example TLE lines
    line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
    line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"

    from astropy import units as u

    vis = Visibility(
        line1, line2, moon_min=20 * u.deg, earthlimb_min=10 * u.deg, sun_min=90 * u.deg,
        jupiter_min=0 * u.deg, mars_min=0 * u.deg
    )

    from astropy.time import Time, TimeDelta

    tstart = Time("2025-01-01T00:00:00.000")
    tstop = Time("2025-01-02T00:00:00.000")  # Example stop time

    import numpy as np

    dt = TimeDelta((1 / 144) * u.day, format="jd")  # 10 minutes in Julian days
    n_steps = int((tstop - tstart) / dt)  # Number of time steps
    time_deltas = TimeDelta(np.arange(n_steps) * dt.jd, format='jd')
    times = tstart + time_deltas

    # using Capella because it is visible in the time period
    from astropy.coordinates import SkyCoord

    target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")
    targ_vis = vis.get_visibility(target_coord, times)

    assert int(targ_vis.shape[0]) == 144
    assert targ_vis.astype(int).sum() == 85


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
    line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"
    
    vis = Visibility(line1, line2)
    target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")
    time = Time("2025-01-01T00:00:00")
    
    # Test with zero constraints
    vis_zero = Visibility(line1, line2, moon_min=0*u.deg, sun_min=0*u.deg, 
                         earthlimb_min=-90*u.deg)
    result = vis_zero.get_visibility(target_coord, time)
    assert isinstance(result, bool)
    assert result == True
    
    # Test with very high constraints (should always fail)
    vis_high = Visibility(line1, line2, moon_min=180*u.deg)
    result = vis_high.get_visibility(target_coord, time)
    assert result == False