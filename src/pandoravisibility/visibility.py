import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import TEME, AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec

__all__ = ["Visibility"]


class Visibility:
    """A class to handle Two-Line Element (TLE) data and target visibility."""

    MOON_MIN = 25 * u.deg  # Minimum allowable moon distance
    SUN_MIN = 91 * u.deg  # Minimum allowable sun distance
    EARTHLIMB_MIN = 20 * u.deg  # Minimum allowable Earth limb distance

    def __init__(self, line1: str, line2: str):
        """
        Initialize the TLE object with the two lines of TLE data.

        Parameters:
        line1 : str
            The first line of the TLE.
        line2 : str
            The second line of the TLE.
        """
        self.tle = Satrec.twoline2rv(line1, line2)

    def __repr__(self) -> str:
        """Return a string representation of the TLE object for debugging."""
        return f"<TLE: {self.tle.satnum}>"

    def get_period(self) -> u.Quantity:
        """
        Calculate the orbital period at the epoch of the TLE.

        Returns:
        u.Quantity
            The orbital period in minutes.
        """
        return (2 * np.pi / self.tle.no) * u.minute

    def get_state(self, time: Time = None) -> SkyCoord:
        """
        Calculate position and velocity coordinates at a given time.

        Parameters:
        time : astropy.time.Time, optional
            The time at which to calculate the coordinates. If not provided,
            `self.time` is used if it exists.

        Returns:
        SkyCoord
            The ITRS coordinates and velocities at the given time.
        """
        if time is None:
            if not hasattr(self, "time"):
                raise ValueError(
                    "No time parameter specified and self.time is not defined."
                )
            time = self.time

        # Handle scalar and array-shaped times
        shape = time.shape
        time = time.ravel()
        time = time.utc

        # Compute satellite position and velocity using SGP4
        e, xyz, vxyz = self.tle.sgp4_array(time.jd1, time.jd2)
        x, y, z = xyz.T
        vx, vy, vz = vxyz.T

        # Handle SGP4 errors
        errors = e[e != 0]
        if errors.size > 0:
            raise RuntimeError(SGP4_ERRORS[errors[0]])

        # Construct SkyCoord in TEME frame and convert to ITRS
        state = SkyCoord(
            x=x * u.km,
            y=y * u.km,
            z=z * u.km,
            v_x=vx * u.km / u.s,
            v_y=vy * u.km / u.s,
            v_z=vz * u.km / u.s,
            frame=TEME(obstime=time),
        ).itrs

        # Restore original shape if necessary
        return state.reshape(shape) if shape else state[0]

    def get_constraint(self, target_coord: SkyCoord, body: str, time: Time) -> bool:
        """
        Calculate whether the constraint for the specified body is met.

        Parameters:
        target_coord : SkyCoord
            The target coordinate to compare with.
        body : str
            The celestial body (e.g., "moon", "sun", "earthlimb").
        time : astropy.time.Time
            The time at which to calculate the constraint.

        Returns:
        bool
            True if the constraint is satisfied, False otherwise.
        """
        # Map body names to the corresponding minimum separation
        body_min_map = {
            "moon": self.MOON_MIN,
            "sun": self.SUN_MIN,
            "earthlimb": self.EARTHLIMB_MIN,
        }

        if body not in body_min_map:
            raise ValueError(
                f"Invalid body: {body}. Choose from: {', '.join(body_min_map.keys())}."
            )

        min_angle = body_min_map[body]

        # Calculate observer's geocentric position
        self.time = time
        state = self.get_state()
        observer_location = EarthLocation.from_geocentric(state.x, state.y, state.z)

        if body in ["moon", "sun"]:
            # Compute angular separation between the body and the target
            body_coord = get_body(body, time=self.time, location=observer_location)
            return (
                body_coord.separation(target_coord, origin_mismatch="ignore")
                >= min_angle
            )

        elif body == "earthlimb":
            # Calculate angular distance from the Earth's limb
            return (
                self._get_angle_from_earth_limb(
                    observer_location, target_coord, self.time
                )
                >= min_angle
            )

    def get_visibility(self, target_coord: SkyCoord, time: Time) -> bool:
        """
        Calculate whether the target is visible based on constraints.

        Parameters:
        target_coord : SkyCoord
            The target coordinate to compare with.
        time : astropy.time.Time
            The time at which to calculate the constraint.

        Returns:
        bool
            True if the target is visible, False otherwise.
        """

        self.time = time
        state = self.get_state()
        observer_location = EarthLocation.from_geocentric(state.x, state.y, state.z)

        moon_coord = get_body("moon", time=self.time, location=observer_location)
        moon_vis = (
            moon_coord.separation(target_coord, origin_mismatch="ignore")
            >= self.MOON_MIN
        )

        sun_coord = get_body("sun", time=self.time, location=observer_location)
        sun_vis = (
            sun_coord.separation(target_coord, origin_mismatch="ignore")
            >= self.SUN_MIN
        )

        earthlimb_vis = (
            self._get_angle_from_earth_limb(observer_location, target_coord, self.time)
            >= self.EARTHLIMB_MIN
        )

        visibility = moon_vis * sun_vis * earthlimb_vis
        return visibility

    @staticmethod
    def _get_angle_from_earth_limb(
        observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
    ) -> u.Quantity:
        """
        Calculate the angular distance from the Earth's limb to the target.

        Parameters:
        observer_location : EarthLocation
            The observer's location on the Earth.
        target_coord : SkyCoord
            The target coordinate to compute the angle for.
        obstime : Time
            The observation time.

        Returns:
        u.Quantity
            The angular distance in degrees.
        """
        # Convert target coordinate to the AltAz frame
        altaz = target_coord.transform_to(
            AltAz(location=observer_location, obstime=obstime)
        )
        alt = altaz.alt

        # Calculate the angular radius of the Earth's limb
        x, y, z = observer_location.geocentric
        observer_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        with np.errstate(invalid="ignore"):
            limb_angle = np.arccos(R_earth / observer_distance)

        return alt + limb_angle
