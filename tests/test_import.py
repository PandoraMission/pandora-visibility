from pandoravisibility import Visibility


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
    # Example TLE lines (replace with actual TLE data)
    line1 = "1 99152U 25037A   25216.00000000 .000000000  00000+0  00000-0 0   427"
    line2 = "2 99152  97.7015  44.6980 0000010   0.1045   0.0000 14.89350717  1230"

    vis = Visibility(line1, line2)

    from astropy.time import Time, TimeDelta
    tstart = Time("2025-01-01T00:00:00.000")
    tstop = Time("2025-01-02T00:00:00.000")  # Example stop time

    import numpy as np
    from astropy import units as u
    dt = TimeDelta((1 / 144) * u.day, format="jd")  # 10 minute in Julian days
    times = tstart + np.arange(0, (tstop - tstart).jd, dt.jd)  # Create time array

    # using Capella because it is visible in the time period
    from astropy.coordinates import SkyCoord
    target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")
    targ_vis = vis.get_visibility(target_coord, times)

    assert int(targ_vis.shape[0]) == 144
    assert targ_vis.astype(int).sum() == 85






