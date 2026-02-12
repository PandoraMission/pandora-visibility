import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import TEME, AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec

__all__ = ["Visibility"]


class Visibility:
    """
    A class to handle Two-Line Element (TLE) data and target visibility.

    This class provides functionality to:
    - Calculate satellite positions from TLE data
    - Determine visibility of astronomical targets based on constraints
    - Analyze observation windows and duty cycles
    - Support visualization of visibility data

    Examples:
    ---------
    >>> # Initialize with TLE data
    >>> vis = Visibility(line1, line2)
    >>>
    >>> # Check visibility for a single target and time
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> target = SkyCoord(ra=79.17, dec=45.99, unit="deg")
    >>> time = Time("2026-01-15T00:00:00")
    >>> is_visible = vis.get_visibility(target, time)
    >>> print(vis.summary(target, time))
    >>>
    >>> # Analyze visibility over a time period
    >>> times = Time("2026-01-01") + np.arange(365) * u.day
    >>> visibility = vis.get_visibility(target, times)
    >>>
    >>> # Plot visibility timeline
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(12,4))
    >>> plt.plot(times.utc, visibility)
    >>> plt.xlabel("Time")
    >>> plt.ylabel("Visibility")

    See Also:
    ---------
    sgp4.api.Satrec : Low-level access to SGP4 propagator
    """

    # Default constants - can be overridden per instance
    MOON_MIN = 25 * u.deg
    SUN_MIN = 91 * u.deg
    EARTHLIMB_MIN = 20 * u.deg
    MARS_MIN = 0 * u.deg
    JUPITER_MIN = 0 * u.deg

    # Star tracker keep-out defaults (0 = disabled)
    ST_SUN_MIN = 0 * u.deg
    ST_MOON_MIN = 0 * u.deg
    ST_EARTHLIMB_MIN = 0 * u.deg
    ST_REQUIRED = 1  # Number of star trackers required to pass (0, 1, or 2)

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

        # Star tracker limits
        self.st_sun_min = custom_limits.get("st_sun_min", self.ST_SUN_MIN)
        self.st_moon_min = custom_limits.get("st_moon_min", self.ST_MOON_MIN)
        self.st_earthlimb_min = custom_limits.get(
            "st_earthlimb_min", self.ST_EARTHLIMB_MIN
        )
        self.st_required = custom_limits.get("st_required", self.ST_REQUIRED)
        if self.st_required not in (0, 1, 2):
            raise ValueError(f"st_required must be 0, 1, or 2, got {self.st_required}")

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
        if self.st_sun_min > 0 * u.deg:
            constraints.append(f"st_sun≥{self.st_sun_min:.0f}")
        if self.st_moon_min > 0 * u.deg:
            constraints.append(f"st_moon≥{self.st_moon_min:.0f}")
        if self.st_earthlimb_min > 0 * u.deg:
            constraints.append(f"st_limb≥{self.st_earthlimb_min:.0f}")
        if self._st_constraint_active:
            constraints.append(f"st_req={self.st_required}")

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
        target_coord : SkyCoord or list of SkyCoord
            The target coordinate(s) to compare with. If a list is provided,
            visibility is computed for each target independently and an array
            of results is returned.
        time : astropy.time.Time
            The time at which to calculate the constraint. Can be scalar or array.

        Returns:
        --------
        bool or np.ndarray
            True if the target is visible, False otherwise.
            - Scalar coord + scalar time → bool
            - Scalar coord + array time (M,) → np.ndarray of bool, shape (M,)
            - N coords (list or array) + scalar time → np.ndarray of bool, shape (N,)
            - N coords (list or array) + array time (M,) → np.ndarray of bool, shape (N, M)
        """
        # Handle multiple target coordinates (list or array SkyCoord)
        # Each target defines a different boresight, so must be evaluated independently
        if isinstance(target_coord, list):
            return np.array([self.get_visibility(tc, time) for tc in target_coord])
        if hasattr(target_coord, "shape") and target_coord.shape != ():
            return np.array(
                [self.get_visibility(target_coord[i], time) for i in range(len(target_coord))]
            )

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

        # Apply star tracker constraints
        st_result = self.get_star_tracker_constraint(target_coord, time)
        result = result & st_result

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

    @property
    def _st_constraint_active(self) -> bool:
        """Whether any star tracker constraints are active."""
        return self.st_required > 0 and (
            self.st_sun_min > 0 * u.deg
            or self.st_moon_min > 0 * u.deg
            or self.st_earthlimb_min > 0 * u.deg
        )

    @property
    def _st_checks(self) -> list:
        """Active star tracker constraint checks as (name, limit, key) tuples."""
        checks = []
        if self.st_sun_min > 0 * u.deg:
            checks.append(("sun", self.st_sun_min, "sun_angle"))
        if self.st_moon_min > 0 * u.deg:
            checks.append(("moon", self.st_moon_min, "moon_angle"))
        if self.st_earthlimb_min > 0 * u.deg:
            checks.append(("limb", self.st_earthlimb_min, "earthlimb_angle"))
        return checks

    @staticmethod
    def _get_star_tracker_body_xyz(tracker: int) -> tuple:
        """
        Get the star tracker boresight direction in body frame coordinates.

        Parameters:
        tracker : int
            Star tracker number (1 or 2)

        Returns:
        tuple
            (x, y, z) unit vector in body frame
        """
        if tracker == 1:
            vec = np.array([0.6804, -0.7071, -0.1923], dtype=float)
        elif tracker == 2:
            vec = np.array([0.6804, 0.7071, -0.1923], dtype=float)
        else:
            raise ValueError(f"Invalid tracker number: {tracker}. Must be 1 or 2.")

        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("Star tracker boresight vector has zero magnitude.")

        vec_unit = vec / norm
        return tuple(vec_unit.tolist())

    def get_star_tracker_angles(
        self, target_coord: SkyCoord, time: Time, tracker: int = 1
    ) -> dict:
        """
        Calculate the star tracker sun and Earth angles.

        The payload +Z points at the science target.
        The payload +Y is the cross product of +Z and the sun vector.
        The payload +X completes the right-handed coordinate system.

        Parameters:
        target_coord : SkyCoord
            The science target coordinate (+Z direction)
        time : Time
            The observation time (scalar or array)
        tracker : int
            Star tracker number (1 or 2)

        Returns:
        dict
            Dictionary with 'ra', 'dec', 'sun_angle', 'moon_angle',
            'earth_angle', and 'earthlimb_angle' as Quantities in degrees.
            Values are scalar or array depending on time input.

        Raises:
        ValueError
            If target is too close to the sun (degenerate attitude)
        """
        observer_location = self._get_observer_location(time)

        # Reuse _get_star_tracker_skycoord for the boresight direction
        st_coord = self._get_star_tracker_skycoord(target_coord, time, tracker)

        # Sun angle
        sun_coord = get_body("sun", time=time, location=observer_location)
        sun_gcrs = sun_coord.transform_to("gcrs")
        sun_angle = st_coord.separation(sun_gcrs)

        # Moon angle
        moon_coord = get_body("moon", time=time, location=observer_location)
        moon_gcrs = moon_coord.transform_to("gcrs")
        moon_angle = st_coord.separation(moon_gcrs)

        # Earth center angle (nadir direction)
        obs_gcrs = observer_location.get_gcrs(obstime=time)
        obs_eci = obs_gcrs.cartesian.xyz
        if time.isscalar:
            earth_eci = -obs_eci / np.linalg.norm(obs_eci)
        else:
            earth_eci = -obs_eci / np.linalg.norm(obs_eci, axis=0, keepdims=True)
        earth_coord = SkyCoord(
            x=earth_eci.value[0],
            y=earth_eci.value[1],
            z=earth_eci.value[2],
            representation_type="cartesian",
            frame="gcrs",
        )
        earth_angle = st_coord.separation(earth_coord)

        # Earth limb angle
        earthlimb_angle = self._get_angle_from_earth_limb(
            observer_location, st_coord, time
        )

        return {
            "ra": st_coord.spherical.lon.to(u.deg),
            "dec": st_coord.spherical.lat.to(u.deg),
            "sun_angle": sun_angle.to(u.deg),
            "moon_angle": moon_angle.to(u.deg),
            "earth_angle": earth_angle.to(u.deg),
            "earthlimb_angle": earthlimb_angle.to(u.deg),
        }

    def _get_star_tracker_skycoord(
        self, target_coord: SkyCoord, time: Time, tracker: int
    ) -> SkyCoord:
        """
        Calculate the sky coordinate where a star tracker boresight points.

        The payload +Z points at the science target. The payload +Y is the
        cross product of +Z and the sun vector. The payload +X completes the
        right-handed coordinate system. The star tracker body-frame vector is
        then rotated into the ECI frame.

        Parameters:
        -----------
        target_coord : SkyCoord
            The science target coordinate (+Z payload direction)
        time : Time
            The observation time (scalar or array)
        tracker : int
            Star tracker number (1 or 2)

        Returns:
        --------
        SkyCoord
            The GCRS coordinate of the star tracker boresight

        Raises:
        -------
        ValueError
            If target is aligned with the sun (scalar time only)
        """
        observer_location = self._get_observer_location(time)

        # Target direction unit vector (constant)
        target_gcrs = target_coord.transform_to("gcrs")
        z_payload = target_gcrs.cartesian.xyz.value
        z_payload = z_payload / np.linalg.norm(z_payload)

        # Sun direction unit vector (time-varying)
        sun_coord = get_body("sun", time=time, location=observer_location)
        sun_gcrs = sun_coord.transform_to("gcrs")
        sun_xyz = sun_gcrs.cartesian.xyz.value

        st_body = np.array(self._get_star_tracker_body_xyz(tracker))

        if time.isscalar:
            sun_vec = sun_xyz / np.linalg.norm(sun_xyz)

            y_payload = np.cross(z_payload, sun_vec)
            y_norm = np.linalg.norm(y_payload)
            if y_norm < 1e-10:
                raise ValueError("Cannot determine attitude: target aligned with sun")
            y_payload = y_payload / y_norm
            x_payload = np.cross(y_payload, z_payload)
            x_payload = x_payload / np.linalg.norm(x_payload)

            R = np.column_stack([x_payload, y_payload, z_payload])
            st_eci = R @ st_body
            st_eci = st_eci / np.linalg.norm(st_eci)

            return SkyCoord(
                x=st_eci[0],
                y=st_eci[1],
                z=st_eci[2],
                representation_type="cartesian",
                frame="gcrs",
            )
        else:
            # Array case: sun_xyz shape is (3, N)
            sun_vec = sun_xyz / np.linalg.norm(sun_xyz, axis=0, keepdims=True)
            N = len(time)
            z_payload = np.tile(z_payload.reshape(3, 1), (1, N))

            y_payload = np.cross(z_payload, sun_vec, axis=0)
            y_norms = np.linalg.norm(y_payload, axis=0, keepdims=True)

            # Detect degenerate timesteps where target is aligned with sun
            degenerate = (y_norms < 1e-10).ravel()

            # Safe-divide: set degenerate norms to 1 to avoid division by zero,
            # then overwrite those columns with NaN so they propagate cleanly
            y_norms_safe = np.where(y_norms < 1e-10, 1.0, y_norms)
            y_payload = y_payload / y_norms_safe

            x_payload = np.cross(y_payload, z_payload, axis=0)
            x_norms = np.linalg.norm(x_payload, axis=0, keepdims=True)
            x_norms_safe = np.where(x_norms < 1e-10, 1.0, x_norms)
            x_payload = x_payload / x_norms_safe

            # Transform star tracker body vector to ECI for each timestep
            st_eci = (
                x_payload * st_body[0] + y_payload * st_body[1] + z_payload * st_body[2]
            )
            st_eci = st_eci / np.linalg.norm(st_eci, axis=0, keepdims=True)

            # Mark degenerate timesteps with NaN so downstream separations
            # return NaN (which compares as False against any threshold)
            st_eci[:, degenerate] = np.nan

            return SkyCoord(
                x=st_eci[0],
                y=st_eci[1],
                z=st_eci[2],
                representation_type="cartesian",
                frame="gcrs",
            )

    def get_star_tracker_constraint(self, target_coord: SkyCoord, time: Time):
        """
        Check if the required number of star trackers satisfy all keep-out constraints.

        Evaluates sun, moon, and Earth limb keep-out angles for both star
        trackers and returns True if self.st_required trackers meet all active
        constraints (0 = disabled, 1 = at least one, 2 = both).

        Parameters:
        -----------
        target_coord : SkyCoord
            The science target coordinate
        time : Time
            The observation time (scalar or array)

        Returns:
        --------
        bool or np.ndarray
            True if the required number of star trackers meet all constraints
        """
        if not self._st_constraint_active:
            if time.isscalar:
                return True
            return np.ones(time.shape, dtype=bool)

        checks = self._st_checks
        tracker_results = []

        for tracker in [1, 2]:
            try:
                angles = self.get_star_tracker_angles(target_coord, time, tracker)
            except ValueError:
                if time.isscalar:
                    tracker_results.append(False)
                else:
                    tracker_results.append(np.zeros(time.shape, dtype=bool))
                continue

            if time.isscalar:
                tracker_ok = True
            else:
                tracker_ok = np.ones(time.shape, dtype=bool)

            for _, limit, key in checks:
                tracker_ok = tracker_ok & (angles[key] >= limit)

            tracker_results.append(tracker_ok)

        # Combine per-tracker results based on st_required
        if self.st_required == 1:
            return tracker_results[0] | tracker_results[1]
        else:
            return tracker_results[0] & tracker_results[1]

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

        if self._st_constraint_active:
            constraints["star_tracker"] = self.get_star_tracker_constraint(
                target_coord, time
            )

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
        return separations

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
            "Visibility Summary",
            f"Target: {coord_str}",
            f"Time:   {time.iso}",
            f"Sat:    {self.tle.satnum}",
            "=" * 60,
        ]

        for body in constraints:
            if body == "star_tracker":
                continue  # handled in dedicated section below
            status = "PASS" if constraints[body] else "FAIL"
            status_symbol = "✓" if constraints[body] else "✗"
            min_sep = getattr(self, f"{body}_min")
            actual_sep = separations[body]

            lines.append(
                f"{body.capitalize():<10} {status_symbol} {status:<4} "
                f"(req: {min_sep:>6.1f}, actual: {actual_sep:>6.1f})"
            )

        # Star tracker constraints section
        if self._st_constraint_active:
            lines.append("-" * 60)
            req_label = "both" if self.st_required == 2 else "≥1"
            lines.append(
                f"Star Tracker Constraints (need {req_label} tracker passing):"
            )

            for tracker in [1, 2]:
                try:
                    angles = self.get_star_tracker_angles(target_coord, time, tracker)
                    tracker_pass = True
                    details = []
                    for name, limit, key in self._st_checks:
                        actual = angles[key]
                        ok = bool(actual >= limit)
                        tracker_pass = tracker_pass and ok
                        sym = "✓" if ok else "✗"
                        details.append(
                            f"{name}:{sym} req:{limit:>6.1f} act:{actual:>6.1f}"
                        )
                    symbol = "✓" if tracker_pass else "✗"
                    status = "PASS" if tracker_pass else "FAIL"
                    lines.append(f"  ST{tracker:<8}{symbol} {status}")
                    for d in details:
                        lines.append(f"    {d}")
                except ValueError as e:
                    lines.append(f"  ST{tracker:<8}✗ ERROR ({e})")

            st_combined = self.get_star_tracker_constraint(target_coord, time)
            st_sym = "✓" if st_combined else "✗"
            st_stat = "PASS" if st_combined else "FAIL"
            lines.append(f"  {'Result':<9}{st_sym} {st_stat}")

        overall_status = (
            "VISIBLE" if self.get_visibility(target_coord, time) else "NOT VISIBLE"
        )
        overall_symbol = "✓" if overall_status == "VISIBLE" else "✗"

        lines.extend(["=" * 60, f"Overall: {overall_symbol} {overall_status}"])

        return "\n".join(lines)
