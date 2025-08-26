import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import TEME, AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec

__all__ = ["Visibility"]


class Visibility:
    """A class to handle Two-Line Element (TLE) data and target visibility."""

    # Default constants - can be overridden per instance
    MOON_MIN = 25 * u.deg
    SUN_MIN = 91 * u.deg
    EARTHLIMB_MIN = 20 * u.deg
    MARS_MIN = 0 * u.deg
    JUPITER_MIN = 0 * u.deg

    def __init__(self, line1: str, line2: str, **custom_limits):
        """
        Initialize the TLE object with the two lines of TLE data.

        Parameters:
        line1 : str
            The first line of the TLE.
        line2 : str
            The second line of the TLE.
        **custom_limits : dict
            Optional custom limits (e.g., moon_min=30*u.deg)
        """
        # Validate TLE lines
        if not line1 or not line2:
            raise ValueError("TLE lines cannot be empty")

        try:
            self.tle = Satrec.twoline2rv(line1, line2)
        except Exception as e:
            raise ValueError(f"Invalid TLE data: {e}")

        # Set instance limits (use class defaults if not provided)
        self.moon_min = custom_limits.get("moon_min", self.MOON_MIN)
        self.sun_min = custom_limits.get("sun_min", self.SUN_MIN)
        self.earthlimb_min = custom_limits.get("earthlimb_min", self.EARTHLIMB_MIN)
        self.mars_min = custom_limits.get("mars_min", self.MARS_MIN)
        self.jupiter_min = custom_limits.get("jupiter_min", self.JUPITER_MIN)

    def __repr__(self) -> str:
        """Return a string representation of the TLE object for debugging."""
        constraints = []
        if self.moon_min > 0 * u.deg:
            constraints.append(f"moon≥{self.moon_min:.0f}")
        if self.sun_min > 0 * u.deg:
            constraints.append(f"sun≥{self.sun_min:.0f}")
        if self.earthlimb_min > 0 * u.deg:
            constraints.append(f"limb≥{self.earthlimb_min:.0f}")
        if self.mars_min > 0 * u.deg:
            constraints.append(f"mars≥{self.mars_min:.0f}")
        if self.jupiter_min > 0 * u.deg:
            constraints.append(f"jupiter≥{self.jupiter_min:.0f}")

        constraint_str = ", ".join(constraints) if constraints else "default"
        return f"<Visibility: SAT{self.tle.satnum} [{constraint_str}]>"

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
            The celestial body (e.g., "moon", "sun", "earthlimb", "mars", "jupiter").
        time : astropy.time.Time
            The time at which to calculate the constraint.

        Returns:
        bool
            True if the constraint is satisfied, False otherwise.
        """
        # Map body names to the corresponding minimum separation
        body_min_map = {
            "moon": self.moon_min,
            "sun": self.sun_min,
            "earthlimb": self.earthlimb_min,
            "jupiter": self.jupiter_min,
            "mars": self.mars_min,
        }

        if body not in body_min_map:
            raise ValueError(
                f"Invalid body: {body}. Choose from: {', '.join(body_min_map.keys())}."
            )

        min_angle = body_min_map[body]

        # Calculate observer's geocentric position
        observer_location = self._get_observer_location(time)

        if body in ["moon", "sun", "mars", "jupiter"]:
            # Compute angular separation between the body and the target
            body_coord = get_body(body, time=time, location=observer_location)
            return (
                body_coord.separation(target_coord, origin_mismatch="ignore")
                >= min_angle
            )

        elif body == "earthlimb":
            # Calculate angular distance from the Earth's limb
            return (
                self._get_angle_from_earth_limb(observer_location, target_coord, time)
                >= min_angle
            )

    def _get_observer_location(self, time: Time) -> EarthLocation:
        """Helper method to get observer location without side effects."""
        # Temporarily set time for state calculation
        original_time = getattr(self, "time", None)
        self.time = time
        try:
            state = self.get_state()
            return EarthLocation.from_geocentric(state.x, state.y, state.z)
        finally:
            if original_time is not None:
                self.time = original_time
            elif hasattr(self, "time"):
                delattr(self, "time")

    def get_visibility(self, target_coord: SkyCoord, time: Time):
        """
        Calculate whether the target is visible based on all constraints.

        Parameters:
        -----------
        target_coord : SkyCoord
            The target coordinate to compare with.
        time : astropy.time.Time
            The time at which to calculate the constraint. Can be scalar or array.

        Returns:
        --------
        bool or np.ndarray
            True if the target is visible, False otherwise.
            Returns array if time is an array.
        """
        # Always check these constraints
        required_constraints = ["moon", "sun", "earthlimb"]

        # Add planetary constraints if enabled
        if self.mars_min > 0 * u.deg:
            required_constraints.append("mars")
        if self.jupiter_min > 0 * u.deg:
            required_constraints.append("jupiter")

        # Start with the first constraint
        result = self.get_constraint(target_coord, required_constraints[0], time)

        # Apply remaining constraints using bitwise AND
        for constraint in required_constraints[1:]:
            constraint_result = self.get_constraint(target_coord, constraint, time)
            result = result & constraint_result

        # Return scalar if input was scalar
        if time.isscalar:
            return bool(result)
        else:
            return result

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

    def get_all_constraints(self, target_coord: SkyCoord, time: Time) -> dict:
        """Get status of all active constraints."""
        constraints = {
            "moon": self.get_constraint(target_coord, "moon", time),
            "sun": self.get_constraint(target_coord, "sun", time),
            "earthlimb": self.get_constraint(target_coord, "earthlimb", time),
        }

        if self.mars_min > 0 * u.deg:
            constraints["mars"] = self.get_constraint(target_coord, "mars", time)

        if self.jupiter_min > 0 * u.deg:
            constraints["jupiter"] = self.get_constraint(target_coord, "jupiter", time)

        return constraints

    def get_separations(self, target_coord: SkyCoord, time: Time) -> dict:
        """Get actual separation angles from all bodies."""
        observer_location = self._get_observer_location(time)
        separations = {}

        for body in ["moon", "sun", "mars", "jupiter"]:
            body_coord = get_body(body, time=time, location=observer_location)
            separations[body] = body_coord.separation(
                target_coord, origin_mismatch="ignore"
            )

        separations["earthlimb"] = self._get_angle_from_earth_limb(
            observer_location, target_coord, time
        )
        return separations  # Add this line!


    def summary(self, target_coord: SkyCoord, time: Time) -> str:
        """
        Get a human-readable summary of visibility constraints.

        Parameters:
        -----------
        target_coord : SkyCoord
            The target coordinate to analyze.
        time : Time
            The observation time. Must be scalar (single time point).

        Returns:
        --------
        str
            Formatted summary of all visibility constraints.

        Raises:
        -------
        ValueError
            If time is not scalar (array inputs not supported).
        """
        # Enforce scalar time restriction
        if not time.isscalar:
            raise ValueError(
                "summary() only supports scalar time inputs. "
                "Use get_visibility() or get_all_constraints() for array inputs."
            )

        try:
            constraints = self.get_all_constraints(target_coord, time)
            separations = self.get_separations(target_coord, time)
        except Exception as e:
            return f"Error calculating visibility: {e}"

        # Better coordinate formatting
        coord_str = target_coord.to_string("hmsdms", precision=1)
        if len(coord_str) > 35:
            coord_str = coord_str[:32] + "..."

        lines = [
            f"Visibility Summary",
            f"Target: {coord_str}",
            f"Time:   {time.iso}",
            f"Sat:    {self.tle.satnum}",
            "=" * 60,
        ]

        for body in constraints:
            status = "PASS" if constraints[body] else "FAIL"
            status_symbol = "✓" if constraints[body] else "✗"
            min_sep = getattr(self, f"{body}_min")
            actual_sep = separations[body]

            lines.append(
                f"{body.capitalize():<10} {status_symbol} {status:<4} "
                f"(req: {min_sep:>6.1f}, actual: {actual_sep:>6.1f})"
            )

        overall_status = (
            "VISIBLE" if self.get_visibility(target_coord, time) else "NOT VISIBLE"
        )
        overall_symbol = "✓" if overall_status == "VISIBLE" else "✗"

        lines.extend(["=" * 60, f"Overall: {overall_symbol} {overall_status}"])

        return "\n".join(lines)
