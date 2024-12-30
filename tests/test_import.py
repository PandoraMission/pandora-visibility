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

    from astropy.constants import R_earth
    from astropy import units as u
    assert (vis.tle.a * u.earthRad - R_earth).to(u.km).value > 596
    assert (vis.tle.a * u.earthRad - R_earth).to(u.km).value < 597
