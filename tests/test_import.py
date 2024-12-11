from pandoravisibility import Visibility

def test_visibility():
    # Example TLE lines (replace with actual TLE data)
    line1 = "1 25544U 98067A   21036.62522304  .00001264  00000-0  31404-4 0  9993"
    line2 = "2 25544  51.6411  12.6452 0001589  96.1237 263.1417 15.49318588339242"

    tle = Visibility(line1, line2)

    # Test get_period method
    period = tle.get_period()
    assert period > 0

    # Test get_state method with a time input
    from astropy.time import Time
    time = Time("2025-01-01T00:00:00")
    state = tle.get_state(time)
    assert state is not None