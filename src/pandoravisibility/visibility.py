import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import GCRS, TEME, AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec

__all__ = ["Visibility"]

_R_EARTH_M = R_earth.to(u.m).value


def _validate_angle(value, name):
    """Raise TypeError if *value* is not an astropy angular Quantity."""
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"{name} must be an astropy Quantity with angular units "
            f"(e.g. {name}={value}*u.deg), got {type(value).__name__}"
        )
    if not value.unit.physical_type == "angle":
        raise u.UnitsError(
            f"{name} must have angular units (e.g. u.deg), "
            f"got {value.unit}"
        )


def _validate_time_quantity(value, name):
    """Raise TypeError if *value* is not an astropy time Quantity."""
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"{name} must be an astropy Quantity with time units "
            f"(e.g. {name}={value}*u.min), got {type(value).__name__}"
        )
    if not value.unit.physical_type == "time":
        raise u.UnitsError(
            f"{name} must have time units (e.g. u.min), "
            f"got {value.unit}"
        )


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
    EARTHLIMB_DAY_MIN = None    # None = use EARTHLIMB_MIN
    EARTHLIMB_NIGHT_MIN = None  # None = use EARTHLIMB_MIN
    TWILIGHT_MARGIN = 0 * u.deg  # 0 = sharp terminator (current behaviour)
    DAYNIGHT_MODE = "subsatellite"  # "limb" = nearest-limb-to-target; "subsatellite" = ground below spacecraft
    MARS_MIN = 0 * u.deg
    JUPITER_MIN = 0 * u.deg

    # Star tracker keep-out defaults (0 = disabled)
    ST_SUN_MIN = 0 * u.deg
    ST_MOON_MIN = 0 * u.deg
    ST_EARTHLIMB_MIN = 0 * u.deg
    ST1_EARTHLIMB_MIN = None  # Per-tracker override (None = use ST_EARTHLIMB_MIN)
    ST2_EARTHLIMB_MIN = None  # Per-tracker override (None = use ST_EARTHLIMB_MIN)
    ST_REQUIRED = 1  # Number of star trackers required to pass (0, 1, or 2)
    ROLL = None  # Spacecraft roll about boresight (None = Maximum solar power)

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

        # Validate units on any user-supplied angle parameters
        _angle_params = [
            "moon_min", "sun_min", "earthlimb_min",
            "earthlimb_day_min", "earthlimb_night_min",
            "twilight_margin",
            "mars_min",
            "jupiter_min", "st_sun_min", "st_moon_min",
            "st_earthlimb_min", "st1_earthlimb_min", "st2_earthlimb_min",
            "roll",
        ]
        for key in _angle_params:
            if key in custom_limits and custom_limits[key] is not None:
                _validate_angle(custom_limits[key], key)

        # Set instance limits (use class defaults if not provided)
        self.moon_min = custom_limits.get("moon_min", self.MOON_MIN)
        self.sun_min = custom_limits.get("sun_min", self.SUN_MIN)
        self.earthlimb_min = custom_limits.get("earthlimb_min", self.EARTHLIMB_MIN)
        self.earthlimb_day_min = custom_limits.get(
            "earthlimb_day_min", self.EARTHLIMB_DAY_MIN
        )
        self.earthlimb_night_min = custom_limits.get(
            "earthlimb_night_min", self.EARTHLIMB_NIGHT_MIN
        )
        self.twilight_margin = custom_limits.get(
            "twilight_margin", self.TWILIGHT_MARGIN
        )
        self.daynight_mode = custom_limits.get(
            "daynight_mode", self.DAYNIGHT_MODE
        )
        if self.daynight_mode not in ("limb", "subsatellite"):
            raise ValueError(
                f"daynight_mode must be 'limb' or 'subsatellite', "
                f"got {self.daynight_mode!r}"
            )
        self.mars_min = custom_limits.get("mars_min", self.MARS_MIN)
        self.jupiter_min = custom_limits.get("jupiter_min", self.JUPITER_MIN)

        # Star tracker limits
        self.st_sun_min = custom_limits.get("st_sun_min", self.ST_SUN_MIN)
        self.st_moon_min = custom_limits.get("st_moon_min", self.ST_MOON_MIN)
        self.st_earthlimb_min = custom_limits.get(
            "st_earthlimb_min", self.ST_EARTHLIMB_MIN
        )
        # Per-tracker Earth limb overrides (None = use shared st_earthlimb_min)
        self.st1_earthlimb_min = custom_limits.get(
            "st1_earthlimb_min", self.ST1_EARTHLIMB_MIN
        )
        self.st2_earthlimb_min = custom_limits.get(
            "st2_earthlimb_min", self.ST2_EARTHLIMB_MIN
        )
        self.st_required = custom_limits.get("st_required", self.ST_REQUIRED)
        if self.st_required not in (0, 1, 2):
            raise ValueError(f"st_required must be 0, 1, or 2, got {self.st_required}")

        # Spacecraft roll angle about boresight (None = Sun-constrained)
        self.roll = custom_limits.get("roll", self.ROLL)
        if self.roll is not None:
            self.roll = self.roll.to(u.deg)

        # One-entry cache for time-dependent quantities reused across calls.
        self._precompute_cache_key = None
        self._precompute_cache_value = None

    def __repr__(self) -> str:
        """Return a string representation of the TLE object for debugging."""
        constraints = []
        if self.moon_min > 0 * u.deg:
            constraints.append(f"moon≥{self.moon_min:.0f}")
        if self.sun_min > 0 * u.deg:
            constraints.append(f"sun≥{self.sun_min:.0f}")
        if self.earthlimb_day_min is not None or self.earthlimb_night_min is not None:
            day_lim = self.earthlimb_day_min if self.earthlimb_day_min is not None else self.earthlimb_min
            night_lim = self.earthlimb_night_min if self.earthlimb_night_min is not None else self.earthlimb_min
            constraints.append(f"limb_day≥{day_lim:.0f}")
            constraints.append(f"limb_night≥{night_lim:.0f}")
            if self.daynight_mode != "subsatellite" or self.twilight_margin > 0 * u.deg:
                constraints.append(f"daynight={self.daynight_mode}")
            if self.twilight_margin > 0 * u.deg:
                constraints.append(f"twilight_margin={self.twilight_margin:.0f}")
        elif self.earthlimb_min > 0 * u.deg:
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
            if self.st1_earthlimb_min is not None or self.st2_earthlimb_min is not None:
                st1_lim = self._st_earthlimb_min_for(1)
                st2_lim = self._st_earthlimb_min_for(2)
                constraints.append(f"st1_limb≥{st1_lim:.0f}")
                constraints.append(f"st2_limb≥{st2_lim:.0f}")
            else:
                constraints.append(f"st_limb≥{self.st_earthlimb_min:.0f}")
        elif self.st1_earthlimb_min is not None or self.st2_earthlimb_min is not None:
            st1_lim = self._st_earthlimb_min_for(1)
            st2_lim = self._st_earthlimb_min_for(2)
            if st1_lim > 0 * u.deg:
                constraints.append(f"st1_limb≥{st1_lim:.0f}")
            if st2_lim > 0 * u.deg:
                constraints.append(f"st2_limb≥{st2_lim:.0f}")
        if self._st_constraint_active:
            constraints.append(f"st_req={self.st_required}")
        if self.roll is not None:
            constraints.append(f"roll={self.roll:.1f}")

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
            limb_angle = self._get_angle_from_earth_limb(
                observer_location, target_coord, time
            )
            # Day/night-aware threshold
            if self.earthlimb_day_min is not None or self.earthlimb_night_min is not None:
                obs_gcrs = observer_location.get_gcrs(obstime=time)
                obs_xyz = obs_gcrs.cartesian.xyz.to(u.m).value
                if time.isscalar:
                    zenith_u = obs_xyz / np.linalg.norm(obs_xyz)
                else:
                    zenith_u = obs_xyz / np.linalg.norm(obs_xyz, axis=0, keepdims=True)
                tgt_gcrs = target_coord.transform_to(GCRS(obstime=time))
                tgt_xyz = tgt_gcrs.cartesian.xyz.value
                if time.isscalar:
                    tgt_u = tgt_xyz / np.linalg.norm(tgt_xyz)
                else:
                    tgt_u = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)
                sun_body = get_body("sun", time=time, location=observer_location)
                sun_xyz = sun_body.cartesian.xyz.value
                if time.isscalar:
                    sun_u = sun_xyz / np.linalg.norm(sun_xyz)
                else:
                    sun_u = sun_xyz / np.linalg.norm(sun_xyz, axis=0, keepdims=True)
                # Compute limb half-angle for surface-normal sunlit check
                obs_dist = np.linalg.norm(obs_xyz, axis=0)
                with np.errstate(invalid="ignore"):
                    la_rad = np.arccos(_R_EARTH_M / obs_dist)
                min_angle = self._effective_earthlimb_min_deg(
                    tgt_u, zenith_u, sun_u, limb_angle_rad=la_rad
                ) * u.deg
            return limb_angle >= min_angle

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

    # ------------------------------------------------------------------
    # Internal fast-path helpers: precompute time-dependent data once,
    # then evaluate each target cheaply using numpy dot products.
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_sep_deg(a, b):
        """Angular separation in degrees between (3,...) unit vectors.

        Supports shapes (3,)·(3,), (3,N)·(3,N), or (3,1)·(3,N).
        """
        dot = np.sum(a * b, axis=0)
        return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))

    @staticmethod
    def _fast_limb_deg(target_unit, zenith_unit, limb_angle_rad):
        """Earth limb angle in degrees via geometric calculation.

        elev = arcsin(dot(target, zenith))  [altitude above local horizon]
        limb = arccos(R_earth / observer_dist)
        result = elev + limb
        """
        dot = np.sum(target_unit * zenith_unit, axis=0)
        elev = np.arcsin(np.clip(dot, -1.0, 1.0))
        return np.rad2deg(elev + limb_angle_rad)

    @staticmethod
    def _earthlimb_is_sunlit(target_unit, zenith_unit, sun_unit,
                             limb_angle_rad=None,
                             twilight_margin_deg=0.0):
        """Whether the nearest Earth limb point to the target is sunlit.

        The nearest limb point's outward surface normal is:

            n = cos(limb_angle) * zenith  +  sin(limb_angle) * limb_dir

        where *limb_dir* is the projection of the target direction onto
        the plane perpendicular to the zenith, and *limb_angle* =
        arccos(R_earth / observer_distance).  The limb point is sunlit
        when ``dot(n, sun) > -sin(twilight_margin)``.

        Parameters
        ----------
        target_unit : ndarray, shape (3,) or (3, N)
            Target direction unit vector(s) in GCRS.
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).
        limb_angle_rad : float or ndarray or None
            Earth-limb half-angle in radians (``arccos(R_earth / d)``).
            When *None*, falls back to a simple horizontal projection
            (ignoring the zenith component of the surface normal).
        twilight_margin_deg : float
            Degrees past the geometric terminator to still classify as
            sunlit.  0 (default) reproduces the original sharp
            terminator.  18 is analogous to astronomical twilight.

        Returns
        -------
        bool or ndarray of bool
            True where the nearest limb point is sunlit.
        """
        dot_tz = np.sum(target_unit * zenith_unit, axis=0)
        if target_unit.ndim == 1:
            proj = target_unit - zenith_unit * dot_tz
        else:
            proj = target_unit - zenith_unit * dot_tz[np.newaxis, :]
        proj_norm = np.linalg.norm(proj, axis=0, keepdims=True)
        limb_unit = proj / np.where(proj_norm < 1e-12, 1.0, proj_norm)

        threshold = -np.sin(np.deg2rad(twilight_margin_deg))

        if limb_angle_rad is None:
            # Legacy fallback: horizontal projection only
            return np.sum(limb_unit * sun_unit, axis=0) > threshold

        cos_la = np.cos(limb_angle_rad)
        sin_la = np.sin(limb_angle_rad)
        # Surface normal of the limb point
        dot_n_sun = cos_la * np.sum(zenith_unit * sun_unit, axis=0) + \
                    sin_la * np.sum(limb_unit * sun_unit, axis=0)
        return dot_n_sun > threshold

    @staticmethod
    def _subsatellite_is_sunlit(zenith_unit, sun_unit,
                                twilight_margin_deg=0.0):
        """Whether the subsatellite point (ground below spacecraft) is sunlit.

        The subsatellite point is the point on Earth's surface directly
        below the spacecraft.  It is sunlit when the angle between the
        zenith direction (observer → away from Earth centre) and the Sun
        direction is less than 90° (plus an optional twilight margin).

        Geometrically: ``dot(zenith, sun) > -sin(twilight_margin)``.

        Parameters
        ----------
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).
        twilight_margin_deg : float
            Degrees past the geometric terminator to still classify as
            sunlit.  0 (default) gives a sharp day/night boundary.

        Returns
        -------
        bool or ndarray of bool
            True where the subsatellite point is sunlit.
        """
        threshold = -np.sin(np.deg2rad(twilight_margin_deg))
        dot_zs = np.sum(zenith_unit * sun_unit, axis=0)
        return dot_zs > threshold

    def _effective_earthlimb_min_deg(self, target_unit, zenith_unit, sun_unit,
                                     limb_angle_rad=None):
        """Per-timestep effective Earth limb threshold in degrees.

        When ``earthlimb_day_min`` or ``earthlimb_night_min`` is set,
        returns a scalar or array of thresholds that depend on whether
        the observer is over sunlit or shadowed Earth.  The method used
        to determine day/night is controlled by ``self.daynight_mode``:

        * ``"subsatellite"`` (default): subsatellite point directly below spacecraft.
        * ``"limb"``: nearest limb point to the target direction.

        Otherwise returns a plain scalar from ``earthlimb_min``.

        Parameters
        ----------
        limb_angle_rad : float or ndarray or None
            Earth-limb half-angle in radians, forwarded to
            ``_earthlimb_is_sunlit``.
        """
        if self.earthlimb_day_min is None and self.earthlimb_night_min is None:
            return self.earthlimb_min.to(u.deg).value

        day_deg = (
            self.earthlimb_day_min.to(u.deg).value
            if self.earthlimb_day_min is not None
            else self.earthlimb_min.to(u.deg).value
        )
        night_deg = (
            self.earthlimb_night_min.to(u.deg).value
            if self.earthlimb_night_min is not None
            else self.earthlimb_min.to(u.deg).value
        )

        twilight_deg = self.twilight_margin.to(u.deg).value

        if self.daynight_mode == "subsatellite":
            sunlit = self._subsatellite_is_sunlit(
                zenith_unit, sun_unit,
                twilight_margin_deg=twilight_deg,
            )
        else:
            sunlit = self._earthlimb_is_sunlit(
                target_unit, zenith_unit, sun_unit,
                limb_angle_rad=limb_angle_rad,
                twilight_margin_deg=twilight_deg,
            )
        return np.where(sunlit, day_deg, night_deg)

    def _precompute(self, time: Time) -> dict:
        """Precompute time-dependent quantities shared across targets.

        Everything in the returned dict depends only on the observation
        time(s) and satellite orbit, not on the science target.  Passing
        this dict to ``_get_visibility_single`` avoids redundant SGP4
        propagation, ephemeris lookups, and coordinate transforms.
        """
        # Cache by object identity: common workflows reuse the same Time object
        # across repeated calls (e.g. many target batches on one time grid).
        cache_key = (
            id(time),
            bool(self.mars_min > 0 * u.deg),
            bool(self.jupiter_min > 0 * u.deg),
        )

        if (cache_key == self._precompute_cache_key and
                self._precompute_cache_value is not None):
            return self._precompute_cache_value

        observer_location = self._get_observer_location(time)

        # Observer GCRS position → zenith direction + Earth limb angle
        obs_gcrs = observer_location.get_gcrs(obstime=time)
        obs_xyz = obs_gcrs.cartesian.xyz.to(u.m).value  # (3,) or (3, N)
        if time.isscalar:
            obs_dist = np.linalg.norm(obs_xyz)
            zenith_unit = obs_xyz / obs_dist
        else:
            obs_dist = np.linalg.norm(obs_xyz, axis=0)  # (N,)
            zenith_unit = obs_xyz / obs_dist[np.newaxis, :]  # (3, N)

        with np.errstate(invalid="ignore"):
            limb_angle_rad = np.arccos(_R_EARTH_M / obs_dist)  # scalar or (N,)

        # Body direction unit vectors (normalised cartesian xyz)
        body_units = {}
        for name in ["moon", "sun"]:
            body = get_body(name, time=time, location=observer_location)
            xyz = body.cartesian.xyz.value
            if time.isscalar:
                body_units[name] = xyz / np.linalg.norm(xyz)
            else:
                body_units[name] = xyz / np.linalg.norm(
                    xyz, axis=0, keepdims=True
                )
        if self.mars_min > 0 * u.deg:
            body = get_body("mars", time=time, location=observer_location)
            xyz = body.cartesian.xyz.value
            if time.isscalar:
                body_units["mars"] = xyz / np.linalg.norm(xyz)
            else:
                body_units["mars"] = xyz / np.linalg.norm(
                    xyz, axis=0, keepdims=True
                )
        if self.jupiter_min > 0 * u.deg:
            body = get_body("jupiter", time=time, location=observer_location)
            xyz = body.cartesian.xyz.value
            if time.isscalar:
                body_units["jupiter"] = xyz / np.linalg.norm(xyz)
            else:
                body_units["jupiter"] = xyz / np.linalg.norm(
                    xyz, axis=0, keepdims=True
                )

        pre = {
            "observer_location": observer_location,
            "body_units": body_units,
            "zenith_unit": zenith_unit,
            "limb_angle_rad": limb_angle_rad,
        }
        self._precompute_cache_key = cache_key
        self._precompute_cache_value = pre
        return pre

    def _get_visibility_single(
        self, target_coord: SkyCoord, time: Time, pre: dict,
        effective_roll=None, gcrs_frame=None,
    ):
        """Visibility for one scalar target using precomputed time data."""
        body_units = pre["body_units"]
        zenith_unit = pre["zenith_unit"]
        limb_rad = pre["limb_angle_rad"]

        # Target direction unit vector(s) in GCRS.
        frame = gcrs_frame if gcrs_frame is not None else GCRS(obstime=time)
        tgt_gcrs = target_coord.transform_to(frame)
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        if time.isscalar:
            tgt_unit = tgt_xyz / np.linalg.norm(tgt_xyz)  # (3,)
            tgt_b = tgt_unit
        else:
            tgt_b = tgt_xyz / np.linalg.norm(
                tgt_xyz, axis=0, keepdims=True
            )  # (3, N)
            tgt_unit = tgt_b[:, 0].copy()

        # Boresight body constraints via fast dot-product separation
        moon_deg = self.moon_min.to(u.deg).value
        sun_deg = self.sun_min.to(u.deg).value
        limb_threshold = self._effective_earthlimb_min_deg(
            tgt_b, zenith_unit, body_units["sun"], limb_angle_rad=limb_rad
        )

        result = self._fast_sep_deg(body_units["moon"], tgt_b) >= moon_deg
        result &= self._fast_sep_deg(body_units["sun"], tgt_b) >= sun_deg
        result &= self._fast_limb_deg(tgt_b, zenith_unit, limb_rad) >= limb_threshold

        if self.mars_min > 0 * u.deg:
            result &= (
                self._fast_sep_deg(body_units["mars"], tgt_b)
                >= self.mars_min.to(u.deg).value
            )
        if self.jupiter_min > 0 * u.deg:
            result &= (
                self._fast_sep_deg(body_units["jupiter"], tgt_b)
                >= self.jupiter_min.to(u.deg).value
            )

        # Star tracker constraints
        if self._st_constraint_active:
            st_result = self._get_st_constraint_fast(
                tgt_unit, time, pre, effective_roll=effective_roll,
            )
            result = result & st_result

        if time.isscalar:
            return bool(result)
        return np.asarray(result)

    def _get_st_constraint_fast(self, tgt_unit, time, pre, *,
                                effective_roll=None):
        """Star tracker constraint check using pure numpy.

        Computes the payload attitude matrix once and applies it to both
        tracker boresight vectors.  Angular separations use dot products
        instead of SkyCoord.separation().

        Parameters
        ----------
        tgt_unit : np.ndarray
            Target direction as (3,) unit vector in GCRS.
        time : Time
            Observation time (scalar or array).
        pre : dict
            Precomputed data from ``_precompute()``.
        effective_roll : Quantity or None
            Roll angle to use.  If ``None``, falls back to ``self.roll``.
        """
        roll = effective_roll if effective_roll is not None else self.roll
        body_units = pre["body_units"]
        zenith_unit = pre["zenith_unit"]
        limb_rad = pre["limb_angle_rad"]
        sun_vec = body_units["sun"]

        # Compute payload attitude ONCE for both trackers
        if roll is not None:
            roll_rad = roll.to(u.rad).value
            x_payload, y_payload = self._roll_attitude(tgt_unit, roll_rad)
            if time.isscalar:
                z_col = tgt_unit
            else:
                N = len(time)
                z_col = np.tile(tgt_unit.reshape(3, 1), (1, N))
                x_payload = x_payload[:, np.newaxis]  # (3,) → (3,1)
                y_payload = y_payload[:, np.newaxis]  # (3,) → (3,1)
                degenerate = np.zeros(N, dtype=bool)
        elif time.isscalar:
            y_payload = np.cross(sun_vec, tgt_unit)
            y_norm = np.linalg.norm(y_payload)
            if y_norm < 1e-10:
                return False  # degenerate: both trackers fail
            y_payload = y_payload / y_norm
            x_payload = np.cross(y_payload, tgt_unit)
            x_payload = x_payload / np.linalg.norm(x_payload)
            z_col = tgt_unit
        else:
            N = len(time)
            z_col = np.tile(tgt_unit.reshape(3, 1), (1, N))
            y_payload = np.cross(sun_vec, z_col, axis=0)
            y_norms = np.linalg.norm(y_payload, axis=0, keepdims=True)
            degenerate = (y_norms < 1e-10).ravel()
            y_payload = y_payload / np.where(y_norms < 1e-10, 1.0, y_norms)
            x_payload = np.cross(y_payload, z_col, axis=0)
            x_norms = np.linalg.norm(x_payload, axis=0, keepdims=True)
            x_payload = x_payload / np.where(x_norms < 1e-10, 1.0, x_norms)

        tracker_results = []

        for tracker in [1, 2]:
            checks = self._st_checks_for(tracker)
            st_body = np.array(self._get_star_tracker_body_xyz(tracker))

            # Rotate body-frame vector to ECI
            st_eci = (
                x_payload * st_body[0]
                + y_payload * st_body[1]
                + z_col * st_body[2]
            )

            if time.isscalar:
                st_norm = np.linalg.norm(st_eci)
                if st_norm < 1e-10:
                    tracker_results.append(False)
                    continue
                st_eci = st_eci / st_norm
            else:
                st_eci = st_eci / np.linalg.norm(st_eci, axis=0, keepdims=True)
                st_eci[:, degenerate] = np.nan

            # Check each constraint via dot-product separation
            if time.isscalar:
                tracker_ok = True
            else:
                tracker_ok = np.ones(time.shape, dtype=bool)

            for _, limit, key in checks:
                limit_deg = limit.to(u.deg).value
                if key == "sun_angle":
                    sep = self._fast_sep_deg(st_eci, body_units["sun"])
                elif key == "moon_angle":
                    sep = self._fast_sep_deg(st_eci, body_units["moon"])
                elif key == "earthlimb_angle":
                    sep = self._fast_limb_deg(st_eci, zenith_unit, limb_rad)
                else:
                    continue
                tracker_ok = tracker_ok & (sep >= limit_deg)

            tracker_results.append(tracker_ok)

        if self.st_required == 1:
            combined = tracker_results[0] | tracker_results[1]
        else:
            combined = tracker_results[0] & tracker_results[1]

        if time.isscalar:
            return bool(combined)
        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_visibility(self, target_coord: SkyCoord, time: Time, roll=None):
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
        roll : Quantity, optional
            Spacecraft roll angle about boresight.  Overrides the instance
            ``roll`` for this call only.  ``None`` (default) keeps the
            instance value (which itself defaults to Sun-constrained when
            not set at construction time).

        Returns:
        --------
        bool or np.ndarray
            True if the target is visible, False otherwise.
            - Scalar coord + scalar time → bool
            - Scalar coord + array time (M,) → np.ndarray of bool, shape (M,)
            - N coords (list or array) + scalar time → np.ndarray of bool, shape (N,)
            - N coords (list or array) + array time (M,) → np.ndarray of bool, shape (N, M)
        """
        # Resolve effective roll without mutating instance state
        if roll is not None:
            _validate_angle(roll, "roll")
            effective_roll = roll.to(u.deg)
        else:
            effective_roll = self.roll
        return self._get_visibility_inner(target_coord, time, effective_roll)

    def _get_visibility_inner(self, target_coord: SkyCoord, time: Time,
                               effective_roll=None):
        """Core visibility logic (called by get_visibility).

        Parameters
        ----------
        effective_roll : Quantity or None
            Roll angle to use for this evaluation.  Passed through to
            ``_get_visibility_single`` → ``_get_st_constraint_fast``
            so that instance state is never mutated.
        """
        # Precompute satellite state and body positions once for all targets
        pre = self._precompute(time)
        gcrs_frame = GCRS(obstime=time)

        # Handle multiple target coordinates (list or array SkyCoord)
        # Each target defines a different boresight, so must be evaluated independently
        if isinstance(target_coord, list):
            return np.array(
                [self._get_visibility_single(tc, time, pre, effective_roll,
                                             gcrs_frame)
                 for tc in target_coord]
            )
        if hasattr(target_coord, "shape") and target_coord.shape != ():
            return np.array(
                [
                    self._get_visibility_single(target_coord[i], time, pre,
                                                effective_roll, gcrs_frame)
                    for i in range(len(target_coord))
                ]
            )

        return self._get_visibility_single(target_coord, time, pre,
                                           effective_roll, gcrs_frame)

    def get_visibility_best_roll(
        self, target_coord: SkyCoord, time: Time, roll_step=2 * u.deg,
        orbit_time_step=1 * u.min,
    ) -> dict:
        """
        Calculate visibility using the optimal roll angle for each orbit.

        For each input time, determines which orbit it falls in, finds the
        best fixed roll angle for that orbit by sweeping ``roll_step``-spaced
        angles over a full orbital period sampled every ``orbit_time_step``,
        then evaluates visibility at the specific input time using that roll.

        The best roll is the one satisfying all star-tracker keep-out
        constraints at the greatest number of boresight-visible orbit
        timesteps, with solar array power as tiebreaker.

        Parameters
        ----------
        target_coord : SkyCoord
            The science target coordinate (+Z boresight direction).
        time : Time
            Observation time(s).  Scalar or array.
        roll_step : Quantity, optional
            Roll sweep resolution (default 2 deg).
        orbit_time_step : Quantity, optional
            Time step for the internal orbit sampling used to determine
            the optimal roll (default 1 min).

        Returns
        -------
        dict
            visible : bool or np.ndarray
                True where all constraints (boresight + ST with orbit-best
                roll) pass.
            boresight_visible : bool or np.ndarray
                True where boresight constraints alone pass (before ST/roll).
            roll_deg : float or np.ndarray
                Orbit-optimal roll angle in degrees (NaN where not visible).
                Constant within each orbit.
            n_st_pass : int or np.ndarray
                Number of star trackers passing at the chosen roll (0-2).
            solar_power_frac : float or np.ndarray
                Solar panel power fraction at the chosen roll (NaN if not
                visible).

        Examples
        --------
        >>> vis = Visibility(line1, line2,
        ...                  st_sun_min=44*u.deg,
        ...                  st_earthlimb_min=30*u.deg,
        ...                  st_moon_min=12*u.deg)
        >>> target = SkyCoord(ra=79.17, dec=45.99, unit="deg")
        >>> times = Time("2026-02-15T18:00:00") + np.arange(97) * u.min
        >>> result = vis.get_visibility_best_roll(target, times)
        >>> print(result['visible'].sum(), "visible time steps")
        >>> print("Roll angles used:", result['roll_deg'])
        """
        _validate_angle(roll_step, "roll_step")
        _validate_time_quantity(orbit_time_step, "orbit_time_step")
        if not roll_step.isscalar:
            raise ValueError("roll_step must be a scalar Quantity")
        if not orbit_time_step.isscalar:
            raise ValueError("orbit_time_step must be a scalar Quantity")

        period = self.get_period()
        is_scalar = time.isscalar
        if is_scalar:
            time = Time([time])
        N_input = len(time)

        # Target direction in GCRS for input boresight checks.
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=time))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b_all = tgt_xyz / np.linalg.norm(
            tgt_xyz, axis=0, keepdims=True
        )  # (3, N_input)

        # Roll setup
        step_deg = roll_step.to(u.deg).value
        roll_degs = np.arange(0, 360, step_deg)
        N_roll = len(roll_degs)

        st1_body = np.array(self._get_star_tracker_body_xyz(1))
        st2_body = np.array(self._get_star_tracker_body_xyz(2))
        st1_checks = self._st_checks_for(1)
        st2_checks = self._st_checks_for(2)

        # Output arrays
        out_visible = np.zeros(N_input, dtype=bool)
        out_boresight = np.zeros(N_input, dtype=bool)
        out_roll = np.full(N_input, np.nan)
        out_nst = np.zeros(N_input, dtype=int)
        out_power = np.full(N_input, np.nan)

        # ── Fast path: no ST constraints ───────────────────────────
        if not self._st_constraint_active:
            pre = self._precompute(time)
            bu = pre["body_units"]
            bs = (
                self._fast_sep_deg(bu["moon"], tgt_b_all)
                >= self.moon_min.to(u.deg).value
            )
            bs &= (
                self._fast_sep_deg(bu["sun"], tgt_b_all)
                >= self.sun_min.to(u.deg).value
            )
            bs &= (
                self._fast_limb_deg(
                    tgt_b_all, pre["zenith_unit"], pre["limb_angle_rad"]
                )
                >= self._effective_earthlimb_min_deg(
                    tgt_b_all, pre["zenith_unit"], bu["sun"],
                    limb_angle_rad=pre["limb_angle_rad"]
                )
            )
            if self.mars_min > 0 * u.deg:
                bs &= (
                    self._fast_sep_deg(bu["mars"], tgt_b_all)
                    >= self.mars_min.to(u.deg).value
                )
            if self.jupiter_min > 0 * u.deg:
                bs &= (
                    self._fast_sep_deg(bu["jupiter"], tgt_b_all)
                    >= self.jupiter_min.to(u.deg).value
                )
            bs = np.asarray(bs).ravel()
            out_visible = bs.copy()
            out_boresight = bs.copy()
            if is_scalar:
                return {
                    "visible": bool(out_visible[0]),
                    "boresight_visible": bool(out_boresight[0]),
                    "roll_deg": float(out_roll[0]),
                    "n_st_pass": int(out_nst[0]),
                    "solar_power_frac": float(out_power[0]),
                }
            return {
                "visible": out_visible,
                "boresight_visible": out_boresight,
                "roll_deg": out_roll,
                "n_st_pass": out_nst,
                "solar_power_frac": out_power,
            }

        # ── Group input times into orbits ──────────────────────────
        period_day = period.to(u.day).value
        period_min = period.to(u.min).value
        half_p_min = period_min / 2
        t0_jd = np.min(time.jd)
        dt_day = time.jd - t0_jd
        orbit_id = np.floor(dt_day / period_day).astype(int)

        # Internal orbit sampling parameters
        orb_step_min = orbit_time_step.to(u.min).value
        n_orbit_samp = int(np.ceil(period_min / orb_step_min)) + 1

        for oid in np.unique(orbit_id):
            idx = np.where(orbit_id == oid)[0]
            chunk_times = time[idx]
            chunk_jd = chunk_times.jd
            center = Time(
                (chunk_jd.min() + chunk_jd.max()) / 2, format="jd"
            )

            # Per-orbit representative target direction at orbit center.
            # Aberration shift within one orbit (~97 min) is <0.1",
            # so a single direction is fine for the roll sweep and
            # orbit-sample boresight constraints.
            tgt_gcrs_orb = target_coord.transform_to(GCRS(obstime=center))
            tgt_xyz_orb = tgt_gcrs_orb.cartesian.xyz.value
            tgt_unit = tgt_xyz_orb / np.linalg.norm(tgt_xyz_orb)
            tgt_b = tgt_unit[:, np.newaxis]  # (3, 1) for orbit sampling

            # Per-timestep target directions for input boresight checks
            chunk_tgt_b = tgt_b_all[:, idx]  # (3, N_chunk)

            # ── Find best roll from orbit window ──────────────────
            orbit_times = center + np.linspace(
                -half_p_min, half_p_min, n_orbit_samp
            ) * u.min
            pre_orb = self._precompute(orbit_times)
            bu_orb = pre_orb["body_units"]
            zen_orb = pre_orb["zenith_unit"]
            limb_orb = pre_orb["limb_angle_rad"]

            # Boresight constraints on orbit
            bs_orb = (
                self._fast_sep_deg(bu_orb["moon"], tgt_b)
                >= self.moon_min.to(u.deg).value
            )
            bs_orb &= (
                self._fast_sep_deg(bu_orb["sun"], tgt_b)
                >= self.sun_min.to(u.deg).value
            )
            bs_orb &= (
                self._fast_limb_deg(tgt_b, zen_orb, limb_orb)
                >= self._effective_earthlimb_min_deg(
                    tgt_b, zen_orb, bu_orb["sun"],
                    limb_angle_rad=limb_orb
                )
            )
            if self.mars_min > 0 * u.deg:
                bs_orb &= (
                    self._fast_sep_deg(bu_orb["mars"], tgt_b)
                    >= self.mars_min.to(u.deg).value
                )
            if self.jupiter_min > 0 * u.deg:
                bs_orb &= (
                    self._fast_sep_deg(bu_orb["jupiter"], tgt_b)
                    >= self.jupiter_min.to(u.deg).value
                )
            bs_orb = np.asarray(bs_orb).ravel()

            best_orbit_roll = np.nan

            if self._st_constraint_active and bs_orb.any():
                # Roll sweep over orbit samples
                z_col_orb = np.tile(
                    tgt_unit.reshape(3, 1), (1, n_orbit_samp)
                )
                st1_ok_orb = np.zeros(
                    (N_roll, n_orbit_samp), dtype=bool
                )
                st2_ok_orb = np.zeros(
                    (N_roll, n_orbit_samp), dtype=bool
                )
                solar_orb = np.zeros((N_roll, n_orbit_samp))

                for r, roll_d in enumerate(roll_degs):
                    roll_rad = np.deg2rad(roll_d)
                    x_pay, y_pay = self._roll_attitude(tgt_unit, roll_rad)

                    st1_eci = (
                        x_pay[:, np.newaxis] * st1_body[0]
                        + y_pay[:, np.newaxis] * st1_body[1]
                        + z_col_orb * st1_body[2]
                    )
                    st1_eci = st1_eci / np.linalg.norm(
                        st1_eci, axis=0, keepdims=True
                    )
                    st2_eci = (
                        x_pay[:, np.newaxis] * st2_body[0]
                        + y_pay[:, np.newaxis] * st2_body[1]
                        + z_col_orb * st2_body[2]
                    )
                    st2_eci = st2_eci / np.linalg.norm(
                        st2_eci, axis=0, keepdims=True
                    )

                    t1_ok = np.ones(n_orbit_samp, dtype=bool)
                    for _, limit, key in st1_checks:
                        lim = limit.to(u.deg).value
                        if key == "sun_angle":
                            sep = self._fast_sep_deg(
                                st1_eci, bu_orb["sun"]
                            )
                        elif key == "moon_angle":
                            sep = self._fast_sep_deg(
                                st1_eci, bu_orb["moon"]
                            )
                        elif key == "earthlimb_angle":
                            sep = self._fast_limb_deg(
                                st1_eci, zen_orb, limb_orb
                            )
                        else:
                            continue
                        t1_ok &= sep >= lim
                    st1_ok_orb[r] = t1_ok

                    t2_ok = np.ones(n_orbit_samp, dtype=bool)
                    for _, limit, key in st2_checks:
                        lim = limit.to(u.deg).value
                        if key == "sun_angle":
                            sep = self._fast_sep_deg(
                                st2_eci, bu_orb["sun"]
                            )
                        elif key == "moon_angle":
                            sep = self._fast_sep_deg(
                                st2_eci, bu_orb["moon"]
                            )
                        elif key == "earthlimb_angle":
                            sep = self._fast_limb_deg(
                                st2_eci, zen_orb, limb_orb
                            )
                        else:
                            continue
                        t2_ok &= sep >= lim
                    st2_ok_orb[r] = t2_ok

                    # Solar power
                    cos_sy = np.sum(
                        y_pay[:, np.newaxis] * bu_orb["sun"], axis=0
                    )
                    cos_sy = np.clip(cos_sy, -1.0, 1.0)
                    theta_sy = np.arccos(np.abs(cos_sy))
                    incidence = np.pi / 2 - theta_sy
                    solar_orb[r] = np.cos(incidence)

                # Combined ST requirement
                if self.st_required == 1:
                    st_ok_combined = st1_ok_orb | st2_ok_orb
                else:
                    st_ok_combined = st1_ok_orb & st2_ok_orb

                # Fully visible = boresight AND ST on orbit
                vis_orb = bs_orb[np.newaxis, :] & st_ok_combined
                vis_count = vis_orb.sum(axis=1)  # (N_roll,)
                best_count = vis_count.max()
                if best_count > 0:
                    candidates = np.where(vis_count == best_count)[0]
                    avg_power = np.array([
                        solar_orb[r, vis_orb[r]].mean()
                        for r in candidates
                    ])
                    best_orbit_roll = roll_degs[
                        candidates[np.argmax(avg_power)]
                    ]
                    # Normalize to [-180, 180]
                    best_orbit_roll = (best_orbit_roll + 180) % 360 - 180

            # ── Evaluate at input times with orbit-optimal roll ───
            pre_inp = self._precompute(chunk_times)
            bu_inp = pre_inp["body_units"]
            zen_inp = pre_inp["zenith_unit"]
            limb_inp = pre_inp["limb_angle_rad"]
            N_chunk = len(chunk_times)

            # Boresight at input times (per-timestep target direction)
            bs_inp = (
                self._fast_sep_deg(bu_inp["moon"], chunk_tgt_b)
                >= self.moon_min.to(u.deg).value
            )
            bs_inp &= (
                self._fast_sep_deg(bu_inp["sun"], chunk_tgt_b)
                >= self.sun_min.to(u.deg).value
            )
            bs_inp &= (
                self._fast_limb_deg(chunk_tgt_b, zen_inp, limb_inp)
                >= self._effective_earthlimb_min_deg(
                    chunk_tgt_b, zen_inp, bu_inp["sun"],
                    limb_angle_rad=limb_inp
                )
            )
            if self.mars_min > 0 * u.deg:
                bs_inp &= (
                    self._fast_sep_deg(bu_inp["mars"], chunk_tgt_b)
                    >= self.mars_min.to(u.deg).value
                )
            if self.jupiter_min > 0 * u.deg:
                bs_inp &= (
                    self._fast_sep_deg(bu_inp["jupiter"], chunk_tgt_b)
                    >= self.jupiter_min.to(u.deg).value
                )
            bs_inp = np.asarray(bs_inp).ravel()
            out_boresight[idx] = bs_inp

            if np.isnan(best_orbit_roll):
                # No roll satisfied ST constraints for this orbit;
                # out_visible remains False, out_roll stays NaN.
                continue

            # ST constraints at input times with the orbit-optimal roll
            roll_rad = np.deg2rad(best_orbit_roll)
            x_pay, y_pay = self._roll_attitude(tgt_unit, roll_rad)
            z_col_inp = np.tile(tgt_unit.reshape(3, 1), (1, N_chunk))

            st1_eci = (
                x_pay[:, np.newaxis] * st1_body[0]
                + y_pay[:, np.newaxis] * st1_body[1]
                + z_col_inp * st1_body[2]
            )
            st1_eci = st1_eci / np.linalg.norm(
                st1_eci, axis=0, keepdims=True
            )
            st2_eci = (
                x_pay[:, np.newaxis] * st2_body[0]
                + y_pay[:, np.newaxis] * st2_body[1]
                + z_col_inp * st2_body[2]
            )
            st2_eci = st2_eci / np.linalg.norm(
                st2_eci, axis=0, keepdims=True
            )

            t1_ok = np.ones(N_chunk, dtype=bool)
            for _, limit, key in st1_checks:
                lim = limit.to(u.deg).value
                if key == "sun_angle":
                    sep = self._fast_sep_deg(st1_eci, bu_inp["sun"])
                elif key == "moon_angle":
                    sep = self._fast_sep_deg(st1_eci, bu_inp["moon"])
                elif key == "earthlimb_angle":
                    sep = self._fast_limb_deg(st1_eci, zen_inp, limb_inp)
                else:
                    continue
                t1_ok &= sep >= lim

            t2_ok = np.ones(N_chunk, dtype=bool)
            for _, limit, key in st2_checks:
                lim = limit.to(u.deg).value
                if key == "sun_angle":
                    sep = self._fast_sep_deg(st2_eci, bu_inp["sun"])
                elif key == "moon_angle":
                    sep = self._fast_sep_deg(st2_eci, bu_inp["moon"])
                elif key == "earthlimb_angle":
                    sep = self._fast_limb_deg(st2_eci, zen_inp, limb_inp)
                else:
                    continue
                t2_ok &= sep >= lim

            if self.st_required == 1:
                st_ok_inp = t1_ok | t2_ok
            else:
                st_ok_inp = t1_ok & t2_ok

            vis_inp = bs_inp & st_ok_inp
            out_visible[idx] = vis_inp
            out_nst[idx] = np.where(
                vis_inp, t1_ok.astype(int) + t2_ok.astype(int), 0
            )

            # Solar power at input times
            cos_sy = np.sum(y_pay[:, np.newaxis] * bu_inp["sun"], axis=0)
            cos_sy = np.clip(cos_sy, -1.0, 1.0)
            theta_sy = np.arccos(np.abs(cos_sy))
            incidence = np.pi / 2 - theta_sy
            power = np.cos(incidence)
            out_power[idx] = np.where(vis_inp, power, np.nan)
            out_roll[idx] = np.where(vis_inp, best_orbit_roll, np.nan)

        if is_scalar:
            return {
                "visible": bool(out_visible[0]),
                "boresight_visible": bool(out_boresight[0]),
                "roll_deg": float(out_roll[0]),
                "n_st_pass": int(out_nst[0]),
                "solar_power_frac": float(out_power[0]),
            }
        return {
            "visible": out_visible,
            "boresight_visible": out_boresight,
            "roll_deg": out_roll,
            "n_st_pass": out_nst,
            "solar_power_frac": out_power,
        }

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
        if self.st_required == 0:
            return False
        if self.st_sun_min > 0 * u.deg or self.st_moon_min > 0 * u.deg:
            return True
        # Check if any tracker has an active Earth limb constraint
        for t in [1, 2]:
            if self._st_earthlimb_min_for(t) > 0 * u.deg:
                return True
        return False

    def _st_earthlimb_min_for(self, tracker: int):
        """Effective Earth limb keep-out for a specific tracker.

        Returns the per-tracker override if set, otherwise the shared value.
        """
        if tracker == 1 and self.st1_earthlimb_min is not None:
            return self.st1_earthlimb_min
        elif tracker == 2 and self.st2_earthlimb_min is not None:
            return self.st2_earthlimb_min
        return self.st_earthlimb_min

    def _st_checks_for(self, tracker: int) -> list:
        """Active ST constraint checks for a specific tracker.

        Parameters
        ----------
        tracker : int
            Star tracker number (1 or 2).

        Returns
        -------
        list of (name, limit, key) tuples
        """
        checks = []
        if self.st_sun_min > 0 * u.deg:
            checks.append(("sun", self.st_sun_min, "sun_angle"))
        if self.st_moon_min > 0 * u.deg:
            checks.append(("moon", self.st_moon_min, "moon_angle"))
        limb_min = self._st_earthlimb_min_for(tracker)
        if limb_min > 0 * u.deg:
            checks.append(("limb", limb_min, "earthlimb_angle"))
        return checks

    @property
    def _st_checks(self) -> list:
        """Active star tracker constraint checks using shared limits.

        .. deprecated:: Use ``_st_checks_for(tracker)`` for per-tracker limits.
        """
        return self._st_checks_for(1)  # backward compat: same as tracker 1

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

    @staticmethod
    def _roll_attitude(z_unit, roll_rad):
        """Compute payload X, Y axes from boresight Z and a fixed roll angle.

        The roll angle is measured from the projection of celestial north
        onto the plane perpendicular to the boresight, rotating toward
        ``cross(Z, north_proj)`` (right-hand rule about Z).

        Parameters
        ----------
        z_unit : np.ndarray
            (3,) unit vector along boresight (+Z payload).
        roll_rad : float
            Roll angle in radians.

        Returns
        -------
        x_payload, y_payload : np.ndarray
            (3,) unit vectors for payload +X and +Y axes.
        """
        north = np.array([0.0, 0.0, 1.0])
        north_proj = north - np.dot(north, z_unit) * z_unit
        north_norm = np.linalg.norm(north_proj)
        if north_norm < 1e-8:
            # Boresight near celestial pole — use east as fallback
            east = np.array([1.0, 0.0, 0.0])
            north_proj = east - np.dot(east, z_unit) * z_unit
            north_norm = np.linalg.norm(north_proj)
        x_ref = north_proj / north_norm
        y_ref = np.cross(z_unit, x_ref)
        y_ref = y_ref / np.linalg.norm(y_ref)

        cos_r = np.cos(roll_rad)
        sin_r = np.sin(roll_rad)
        x_payload = cos_r * x_ref + sin_r * y_ref
        y_payload = -sin_r * x_ref + cos_r * y_ref
        return x_payload, y_payload

    def get_star_tracker_angles(
        self, target_coord: SkyCoord, time: Time, tracker: int = 1
    ) -> dict:
        """
        Calculate the star tracker sun and Earth angles.

        When ``roll`` is None (default), the payload attitude is
        Sun-constrained: +Y = Sun × Z, +X = Y × Z.
        When ``roll`` is set, the attitude is determined by rotating
        from the celestial-north projection by that angle about Z.

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
        sun_angle = st_coord.separation(sun_coord)

        # Moon angle
        moon_coord = get_body("moon", time=time, location=observer_location)
        moon_angle = st_coord.separation(moon_coord)

        # Earth center angle (nadir direction).
        # Use the same topocentric frame as st_coord so separation() does not
        # need a frame translation (which would distort nearby unit-distance
        # SkyCoords).
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
            frame=st_coord.frame,
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

        The payload +Z points at the science target.  When ``self.roll``
        is None, the attitude is Sun-constrained (+Y = Sun × Z).  When
        ``self.roll`` is an angle, the attitude is set by rotating from
        celestial-north projection by that roll about Z.  The star
        tracker body-frame vector is then rotated into the ECI frame.

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

        # Satellite GCRS frame (topocentric: obsgeoloc = satellite position).
        # Body SkyCoords from get_body(location=observer_location) carry the
        # same obsgeoloc, so separation() won't apply a spurious origin
        # translation that shifts the Moon direction by up to ~1 deg.
        obs_gcrs = observer_location.get_gcrs(obstime=time)
        sat_gcrs_frame = GCRS(
            obstime=time,
            obsgeoloc=obs_gcrs.cartesian.without_differentials(),
            obsgeovel=obs_gcrs.velocity.d_xyz,
        )

        # Target direction unit vector(s) in GCRS at each observation time.
        # Using obstime=time (not just time[0]) correctly accounts for
        # aberration and precession over long time arrays.
        target_gcrs = target_coord.transform_to(GCRS(obstime=time))
        z_payload_raw = target_gcrs.cartesian.xyz.value

        if time.isscalar:
            z_payload = z_payload_raw / np.linalg.norm(z_payload_raw)
        else:
            z_payload = z_payload_raw / np.linalg.norm(
                z_payload_raw, axis=0, keepdims=True
            )  # (3, N)

        st_body = np.array(self._get_star_tracker_body_xyz(tracker))

        if self.roll is not None:
            # Fixed-roll attitude: no Sun dependency.
            # Use representative direction for attitude frame; the
            # per-timestep z_payload is used for the final rotation.
            roll_rad = self.roll.to(u.rad).value
            z_rep = z_payload if time.isscalar else z_payload[:, 0]
            x_payload, y_payload = self._roll_attitude(z_rep, roll_rad)

            if time.isscalar:
                R = np.column_stack([x_payload, y_payload, z_payload])
                st_eci = R @ st_body
                st_eci = st_eci / np.linalg.norm(st_eci)
            else:
                st_eci = (
                    x_payload[:, np.newaxis] * st_body[0]
                    + y_payload[:, np.newaxis] * st_body[1]
                    + z_payload * st_body[2]
                )
                st_eci = st_eci / np.linalg.norm(st_eci, axis=0, keepdims=True)

            return SkyCoord(
                x=st_eci[0],
                y=st_eci[1],
                z=st_eci[2],
                representation_type="cartesian",
                frame=sat_gcrs_frame,
            )

        # Sun-constrained attitude (default)
        sun_coord = get_body("sun", time=time, location=observer_location)
        sun_xyz = sun_coord.cartesian.xyz.value

        if time.isscalar:
            sun_vec = sun_xyz / np.linalg.norm(sun_xyz)

            y_payload = np.cross(sun_vec, z_payload)
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
                frame=sat_gcrs_frame,
            )
        else:
            # Array case: sun_xyz shape is (3, N)
            sun_vec = sun_xyz / np.linalg.norm(sun_xyz, axis=0, keepdims=True)
            # z_payload is already (3, N) from per-timestep GCRS transform

            y_payload = np.cross(sun_vec, z_payload, axis=0)
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
                frame=sat_gcrs_frame,
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

        tracker_results = []

        for tracker in [1, 2]:
            checks = self._st_checks_for(tracker)
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
            combined = tracker_results[0] | tracker_results[1]
        else:
            combined = tracker_results[0] & tracker_results[1]

        # Normalize scalar result to plain Python bool
        if time.isscalar:
            return bool(combined)
        return combined

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
            actual_sep = separations[body]

            if body == "earthlimb" and (
                self.earthlimb_day_min is not None
                or self.earthlimb_night_min is not None
            ):
                # Show day/night thresholds and which is active
                day_lim = (
                    self.earthlimb_day_min
                    if self.earthlimb_day_min is not None
                    else self.earthlimb_min
                )
                night_lim = (
                    self.earthlimb_night_min
                    if self.earthlimb_night_min is not None
                    else self.earthlimb_min
                )
                # Determine whether limb point is sunlit at this time
                observer_location = self._get_observer_location(time)
                obs_gcrs = observer_location.get_gcrs(obstime=time)
                obs_xyz = obs_gcrs.cartesian.xyz.to(u.m).value
                zenith_u = obs_xyz / np.linalg.norm(obs_xyz)
                tgt_gcrs = target_coord.transform_to(GCRS(obstime=time))
                tgt_xyz = tgt_gcrs.cartesian.xyz.value
                tgt_u = tgt_xyz / np.linalg.norm(tgt_xyz)
                sun_body = get_body("sun", time=time, location=observer_location)
                sun_xyz = sun_body.cartesian.xyz.value
                sun_u = sun_xyz / np.linalg.norm(sun_xyz)
                obs_dist = np.linalg.norm(obs_xyz)
                with np.errstate(invalid="ignore"):
                    la_rad = np.arccos(_R_EARTH_M / obs_dist)
                is_sunlit = bool(self._earthlimb_is_sunlit(
                    tgt_u, zenith_u, sun_u, limb_angle_rad=la_rad,
                    twilight_margin_deg=self.twilight_margin.to(u.deg).value,
                ))
                side = "day" if is_sunlit else "night"
                eff_lim = day_lim if is_sunlit else night_lim
                lines.append(
                    f"{body.capitalize():<10} {status_symbol} {status:<4} "
                    f"(req: {eff_lim:>6.1f} [{side}], actual: {actual_sep:>6.1f})"
                )
            else:
                min_sep = getattr(self, f"{body}_min")
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
                    for name, limit, key in self._st_checks_for(tracker):
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
