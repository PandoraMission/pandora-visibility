"""
Helper functions for ``solar_system_visibility.ipynb``.

Four groups of functions live here:

* resolving planet and moon names to JPL Horizons ids, and each body's system
  (its primary and the primary's major moons);
* JPL Horizons observer ephemerides, fetched over the Horizons API with the
  standard library, including Horizons' moon visibility codes (occulted,
  eclipsed, transiting);
* sampling extended bodies (an oblate planet disk and Saturn's main rings) as
  point sources, so ``pandoravisibility.trace`` can draw and score them;
* the NIRDA field around a target at one time step: the other bodies of its
  system plus Gaia field stars;
* the same for the VISDA (visible camera) field, for ``pandoravisibility.visda``;
* a moon's place in its orbit as seen from Pandora (orbital phase, angle from
  the line of sight through its planet) and where sunlight could glint off
  surface liquid (Titan's seas).

The ephemerides are geocentric. Pandora's ~7,000 km offset from the Earth's
center moves every body of a planetary system by the same ~1 arcsec, so the
relative geometry (which decides occultations and trace overlaps) is unchanged.
"""

import json
import urllib.parse
import urllib.request

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, SkyCoord

from pandoravisibility import star_field, trace

__all__ = [
    "resolve_body",
    "is_primary",
    "system_ids",
    "horizons_ephemeris",
    "system_ephemerides",
    "mean_coord",
    "is_clear",
    "pandora_centric",
    "body_samples",
    "system_field",
    "system_visible_sources",
    "moon_geometry",
    "specular_point",
    "glint_check",
    "fresnel_reflectance",
    "illuminated_disk",
    "PLANETS",
    "DWARF_PLANETS",
    "MAJOR_MOONS",
]

#: JPL Horizons API endpoint.
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

#: Planets by Horizons id of the body itself (x99), not the system barycenter.
PLANETS = {"Mars": 499, "Jupiter": 599, "Saturn": 699, "Uranus": 799, "Neptune": 899}

#: Dwarf planets, by Horizons SPK id; each is its own system (no moons drawn).
DWARF_PLANETS = {"Ceres": 2000001}

#: Horizons COMMAND strings for bodies not looked up by their id. Ceres by its
#: asteroid number, which (unlike its SPK id) also gives its magnitude.
_HORIZONS_COMMANDS = {2000001: "1;"}

#: Major moons per planet, by Horizons id.
MAJOR_MOONS = {
    499: {"Phobos": 401, "Deimos": 402},
    599: {"Io": 501, "Europa": 502, "Ganymede": 503, "Callisto": 504},
    699: {
        "Mimas": 601,
        "Enceladus": 602,
        "Tethys": 603,
        "Dione": 604,
        "Rhea": 605,
        "Titan": 606,
        "Hyperion": 607,
        "Iapetus": 608,
    },
    799: {"Ariel": 701, "Umbriel": 702, "Titania": 703, "Oberon": 704, "Miranda": 705},
    899: {"Triton": 801},
}

#: Polar over equatorial radius, for the shape of a planet's disk.
POLAR_RATIO = {
    499: 0.995,
    599: 0.935,
    699: 0.902,
    799: 0.977,
    899: 0.983,
    2000001: 0.925,  # Ceres, from Dawn (446 / 482 km)
}

#: Main rings drawn, as (inner, outer) radius in planet equatorial radii:
#: Saturn's B ring inner edge (92,000 km) to A ring outer edge (136,775 km),
#: Cassini division not resolved. Equatorial radius 60,268 km.
RINGS = {699: (92000.0 / 60268.0, 136775.0 / 60268.0)}

#: Ring surface brightness relative to the planet's disk, per unit projected
#: area. A placeholder (Saturn's B ring and disk have similar albedos); the
#: planet's total flux is fixed by its Horizons magnitude either way.
RING_RELATIVE_BRIGHTNESS = 1.0

#: The Sun's V - J. Planets and moons reflect sunlight, so their J is taken as
#: V - SOLAR_V_MINUS_J. Methane bands make Saturn, Titan, Uranus and Neptune
#: fainter in the near infrared than this suggests.
SOLAR_V_MINUS_J = 1.15

#: The Sun's Gaia BP - RP and G - V (the Riello et al. 2021 G - V relation at
#: that color), for drawing sunlit bodies on VISDA.
SOLAR_BP_RP = 0.82
SOLAR_G_MINUS_V = -0.15

#: Spacing of the point samples of an extended body, in arcsec (half a pixel).
SAMPLE_SPACING = 0.6

#: Bodies smaller than this (arcsec) are drawn as single points.
EXTENDED_MIN_DIAMETER = 2.0 * trace.NIRDA_PIXEL_SCALE

#: Horizons satellite visibility codes. "Clear" is the only state counted as
#: observable; "-" marks a primary body.
VISIBILITY_CODES = {
    "*": "free and clear",
    "t": "transiting the planet's disk",
    "O": "occulted by the planet",
    "p": "partial umbral eclipse",
    "P": "occulted, partial umbral eclipse",
    "u": "total umbral eclipse",
    "U": "occulted, total umbral eclipse",
    "-": "primary body",
}

#: Codes whose light is hidden (behind the planet or in its full shadow).
HIDDEN_CODES = ("O", "P", "u", "U")

#: Refractive index of Titan's lake liquid in the near infrared: a methane-
#: ethane mix, between liquid methane (~1.27) and liquid ethane (~1.36).
#: Only sets the glint's brightness (``fresnel_reflectance``), not where it is.
TITAN_LIQUID_INDEX = 1.29

#: Surface liquid that can glint, per Horizons id: (name, latitude, west
#: longitude, radius), all in degrees. Each sea or lake is a circle of its
#: area's equivalent radius about its center (USGS nomenclature, rounded);
#: the long seas' edges are only roughly matched.
LIQUID_FEATURES = {
    606: [
        ("Kraken Mare", 68.0, 310.0, 10.0),
        ("Ligeia Mare", 79.0, 248.0, 4.5),
        ("Punga Mare", 85.0, 340.0, 3.0),
        ("Jingpo Lacus", 73.0, 336.0, 1.8),
        ("Ontario Lacus", -72.0, 183.0, 1.5),
    ],
}

# Horizons observer quantities: 1 astrometric RA/Dec, 9 visual magnitude,
# 12 satellite separation and visibility code, 13 angular diameter,
# 14 sub-observer longitude/latitude, 15 sub-solar longitude/latitude,
# 17 north pole position angle, 20 observer range, 24 Sun-target-observer
# (phase) angle.
_QUANTITIES = "1,9,12,13,14,15,17,20,24"
_COLUMNS = {
    "R.A.___(ICRF)": "ra",
    "DEC____(ICRF)": "dec",
    "APmag": "vmag",
    "vis.": "code",
    "Ang-diam": "diameter",
    "ObsSub-LON": "sub_lon",
    "ObsSub-LAT": "sub_lat",
    "SunSub-LON": "sun_lon",
    "SunSub-LAT": "sun_lat",
    "NP.ang": "pole_pa",
    "delta": "delta",
    "S-T-O": "phase_angle",
}

#: Kilometers per astronomical unit (IAU 2012).
_AU_KM = 149597870.7


def resolve_body(name):
    """
    Horizons id and display name of a planet or major moon.

    Parameters
    ----------
    name : str or int
        A name from ``PLANETS``, ``DWARF_PLANETS`` or ``MAJOR_MOONS`` (any case), or a Horizons
        id such as 606.

    Returns
    -------
    body_id : int
    label : str
    """
    names = {**PLANETS, **DWARF_PLANETS}
    for moons in MAJOR_MOONS.values():
        names.update(moons)
    if isinstance(name, (int, np.integer)):
        for label, body_id in names.items():
            if body_id == name:
                return int(body_id), label
        raise ValueError(f"Horizons id {name} is not a planet or major moon here")
    for label, body_id in names.items():
        if label.lower() == str(name).strip().lower():
            return body_id, label
    raise ValueError(
        f"{name!r} is not a known planet, dwarf planet or major moon; choose from "
        + ", ".join(names)
    )


def is_primary(body_id):
    """True for a planet or dwarf planet, False for a moon."""
    return body_id % 100 == 99 or body_id in DWARF_PLANETS.values()


def system_ids(body_id):
    """Ids of a body's system: its primary first, then the primary's moons."""
    primary = body_id if is_primary(body_id) else (body_id // 100) * 100 + 99
    return [primary] + list(MAJOR_MOONS.get(primary, {}).values())


def horizons_ephemeris(body_id, start_utc, stop_utc, step_minutes):
    """
    Geocentric observer ephemeris of one body from JPL Horizons.

    Parameters
    ----------
    body_id : int
        Horizons id.
    start_utc, stop_utc : str
        ISO UTC limits, e.g. ``"2026-10-22T00:00:00"``.
    step_minutes : float
        Step between rows, in whole minutes.

    Returns
    -------
    dict of np.ndarray
        jd (UT), ra and dec (astrometric ICRF, deg), vmag, code (Horizons
        visibility code, see ``VISIBILITY_CODES``), diameter (arcsec),
        sub_lon and sub_lat (sub-observer point, deg; sub_lat is the ring
        opening angle for a ringed planet), sun_lon and sun_lat (sub-solar
        point, deg), pole_pa (north pole position angle, deg east of north),
        delta (distance from the Earth's center, au) and phase_angle
        (Sun-body-Earth angle, deg). Longitudes are in the body's own
        convention (west-positive for Titan and the planets, east-positive
        for Ceres). Values Horizons
        does not provide are NaN.
    """
    params = {
        "format": "json",
        "COMMAND": f"'{_HORIZONS_COMMANDS.get(body_id, body_id)}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "START_TIME": f"'{start_utc.replace('T', ' ')}'",
        "STOP_TIME": f"'{stop_utc.replace('T', ' ')}'",
        "STEP_SIZE": f"'{int(round(step_minutes))} m'",
        "QUANTITIES": f"'{_QUANTITIES}'",
        "ANG_FORMAT": "DEG",
        "CSV_FORMAT": "YES",
        "CAL_FORMAT": "JD",
        # RA/Dec to 1e-9 deg (~0.1 km at Saturn) instead of 1e-5 (~200 km),
        # which moon_geometry needs for the moons' motion about the planet
        "EXTRA_PREC": "YES",
    }
    url = f"{HORIZONS_API_URL}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        text = json.load(response)["result"]
    if "$$SOE" not in text:
        raise RuntimeError(f"Horizons returned no ephemeris for {body_id}:\n{text}")

    header_text, table = text.split("$$SOE")
    table = table.split("$$EOE")[0]
    header = [name.strip() for name in header_text.strip().splitlines()[-2].split(",")]
    rows = [
        [value.strip() for value in line.split(",")]
        for line in table.strip().splitlines()
    ]
    columns = {name: index for index, name in enumerate(header) if name}
    result = {"jd": np.array([float(row[0]) for row in rows])}
    for name, key in _COLUMNS.items():
        values = [row[columns[name]] for row in rows]
        if key == "code":
            result[key] = np.array(values)
        else:
            result[key] = np.array(
                [float(v) if v not in ("", "n.a.") else np.nan for v in values]
            )
    return result


def system_ephemerides(target_ids, start_utc, stop_utc, step_minutes, times):
    """
    Ephemerides of every body in the targets' systems, on the notebook grid.

    Each body is fetched once from Horizons and sampled at ``times``:
    numbers are interpolated, visibility codes taken from the nearest row.

    Returns
    -------
    dict
        Horizons id -> ephemeris dict (see ``horizons_ephemeris``), every
        array with one entry per time in ``times``.
    """
    ephemerides = {}
    for target_id in target_ids:
        for body_id in system_ids(target_id):
            if body_id in ephemerides:
                continue
            raw = horizons_ephemeris(body_id, start_utc, stop_utc, step_minutes)
            after = np.clip(np.searchsorted(raw["jd"], times.jd), 1, len(raw["jd"]) - 1)
            before_closer = (times.jd - raw["jd"][after - 1]) < (
                raw["jd"][after] - times.jd
            )
            nearest = np.where(before_closer, after - 1, after)
            sampled = {"code": raw["code"][nearest]}
            for key in (
                "dec",
                "vmag",
                "diameter",
                "sub_lat",
                "sun_lat",
                "pole_pa",
                "delta",
                "phase_angle",
            ):
                sampled[key] = np.interp(times.jd, raw["jd"], raw[key])
            # Angles that wrap at 360 deg go through their unit vectors
            for key in ("ra", "sub_lon", "sun_lon"):
                angle = np.deg2rad(raw[key])
                sampled[key] = (
                    np.rad2deg(
                        np.arctan2(
                            np.interp(times.jd, raw["jd"], np.sin(angle)),
                            np.interp(times.jd, raw["jd"], np.cos(angle)),
                        )
                    )
                    % 360.0
                )
            ephemerides[body_id] = sampled
    return ephemerides


def mean_coord(ephemeris, indices):
    """Mean sky position of a body over some time steps, as a SkyCoord."""
    ra = np.deg2rad(ephemeris["ra"][indices])
    dec = np.deg2rad(ephemeris["dec"][indices])
    # Average the unit vectors, which handles the 0/360 deg RA wrap
    x = np.mean(np.cos(dec) * np.cos(ra))
    y = np.mean(np.cos(dec) * np.sin(ra))
    z = np.mean(np.sin(dec))
    return SkyCoord(
        np.rad2deg(np.arctan2(y, x)) % 360.0,
        np.rad2deg(np.arctan2(z, np.hypot(x, y))),
        unit="deg",
    )


def pandora_centric(ephemeris, visibility, times):
    """
    A body's position as seen from Pandora rather than the Earth's center.

    Horizons positions here are geocentric; Pandora sits ~7,000 km from the
    Earth's center, which moves a body by ~7,000 km / its distance: about
    1 arcsec for Saturn, up to ~20 arcsec for Mars near opposition. The
    spacecraft position comes from the TLE through ``Visibility.get_state``.
    Like Horizons' positions these are astrometric ICRF (light time
    included, no aberration), the frame star catalog positions are given in.

    Parameters
    ----------
    ephemeris : dict
        The body's sampled ephemeris (see ``system_ephemerides``).
    visibility : pandoravisibility.Visibility
        Supplies the spacecraft orbit.
    times : astropy.time.Time
        The notebook time grid.

    Returns
    -------
    ra, dec : np.ndarray
        Pandora-centric position in degrees, one per time.
    """
    x, y, z = _geocentric_position(ephemeris) - _spacecraft_position(visibility, times)
    return np.rad2deg(np.arctan2(y, x)) % 360.0, np.rad2deg(
        np.arctan2(z, np.hypot(x, y))
    )


def _spacecraft_position(visibility, times):
    """Pandora's geocentric position (km, ICRS axes), shape (3, n)."""
    return (
        visibility.get_state(times)
        .transform_to(GCRS(obstime=times))
        .cartesian.xyz.to_value(u.km)
    )


def _geocentric_position(ephemeris):
    """A body's geocentric position (km, ICRS axes) from RA, Dec and range."""
    ra = np.deg2rad(ephemeris["ra"])
    dec = np.deg2rad(ephemeris["dec"])
    distance = ephemeris["delta"] * _AU_KM
    return distance * np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )


def is_clear(body_id, ephemeris):
    """Per-step flag: True where a moon is free and clear (planets always)."""
    if is_primary(body_id):
        return np.ones(len(ephemeris["code"]), dtype=bool)
    return ephemeris["code"] == "*"


def body_samples(body_id, ephemeris, index):
    """
    Point samples of a body's light on the sky, for ``pandoravisibility.trace``.

    A body smaller than ``EXTENDED_MIN_DIAMETER`` is one point. A larger one
    is a uniform oblate disk (no limb darkening) and, for a planet in
    ``RINGS``, a ring annulus seen at the sub-observer latitude, sampled every
    ``SAMPLE_SPACING``. Ring shadows and the overlap of disk and rings are
    ignored.

    Parameters
    ----------
    body_id : int
        Horizons id.
    ephemeris : dict
        The body's sampled ephemeris (see ``system_ephemerides``).
    index : int
        Time step.

    Returns
    -------
    dxi, deta, weight : np.ndarray
        Offsets from the body's center in arcsec (east, north) and the share
        of its light at each sample, summing to 1. The first sample is the
        one nearest the center.
    """
    diameter = ephemeris["diameter"][index]
    if not np.isfinite(diameter) or diameter < EXTENDED_MIN_DIAMETER:
        return np.zeros(1), np.zeros(1), np.ones(1)

    radius = diameter / 2.0
    opening = np.deg2rad(np.nan_to_num(ephemeris["sub_lat"][index]))
    pole_pa = np.deg2rad(np.nan_to_num(ephemeris["pole_pa"][index]))
    # Apparent flattening of the disk depends on how the pole is tilted to us
    polar = POLAR_RATIO.get(body_id, 1.0)
    apparent_polar = np.sqrt((polar * np.cos(opening)) ** 2 + np.sin(opening) ** 2)

    # Disk: grid in (equator, pole) coordinates
    grid = np.arange(-radius, radius + SAMPLE_SPACING / 2, SAMPLE_SPACING)
    along_equator, along_pole = (axis.ravel() for axis in np.meshgrid(grid, grid))
    inside = (along_equator / radius) ** 2 + (
        along_pole / (radius * apparent_polar)
    ) ** 2 <= 1.0
    along_equator, along_pole = along_equator[inside], along_pole[inside]
    weight = np.ones(along_equator.size)

    if body_id in RINGS:
        # Rings: polar grid in the ring plane, foreshortened along the pole
        inner, outer = (radius * edge for edge in RINGS[body_id])
        ring_radii = np.arange(inner, outer, SAMPLE_SPACING) + SAMPLE_SPACING / 2
        ring_equator, ring_pole, ring_weight = [], [], []
        for ring_radius in ring_radii:
            n_azimuth = max(int(2 * np.pi * ring_radius / SAMPLE_SPACING), 8)
            azimuth = np.arange(n_azimuth) * 2 * np.pi / n_azimuth
            ring_equator.append(ring_radius * np.cos(azimuth))
            ring_pole.append(ring_radius * np.sin(azimuth) * np.sin(opening))
            # Projected area of each sample relative to a disk sample
            ring_weight.append(
                np.full(
                    n_azimuth,
                    RING_RELATIVE_BRIGHTNESS
                    * 2
                    * np.pi
                    * ring_radius
                    * SAMPLE_SPACING
                    / n_azimuth
                    * abs(np.sin(opening))
                    / SAMPLE_SPACING**2,
                )
            )
        along_equator = np.concatenate([along_equator, *ring_equator])
        along_pole = np.concatenate([along_pole, *ring_pole])
        weight = np.concatenate([weight, *ring_weight])

    # To the sky: the pole points to position angle pole_pa (east of north)
    dxi = along_equator * np.cos(pole_pa) + along_pole * np.sin(pole_pa)
    deta = -along_equator * np.sin(pole_pa) + along_pole * np.cos(pole_pa)
    first = np.argmin(np.hypot(dxi, deta))
    order = np.r_[first, np.delete(np.arange(dxi.size), first)]
    return dxi[order], deta[order], weight[order] / weight.sum()


def system_field(target_id, ephemerides, index, time, labels, include_stars=True):
    """
    NIRDA field around a target at one time step, for ``trace.plot_trace``.

    The target sits at the pivot, extended if it is (``target_shape``). The
    other bodies of its system are neighbors, dropped while hidden (occulted
    or fully eclipsed, see ``HIDDEN_CODES``), and Gaia field stars from
    ``star_field.query_stars`` are added. Brightness ratios use J estimated
    as V - ``SOLAR_V_MINUS_J`` for the bodies.

    Parameters
    ----------
    target_id : int
        Horizons id of the body at the pivot.
    ephemerides : dict
        From ``system_ephemerides``.
    index : int
        Time step.
    time : astropy.time.Time
        That time step, for the stars' proper motion.
    labels : dict
        Horizons id -> display name.
    include_stars : bool
        Add Gaia field stars (needs the gaiaoffline database).

    Returns
    -------
    dict
        ``xi``, ``eta``, ``flux_ratio``, ``jmag`` and ``label`` per entry,
        plus ``target_shape``, in the format ``pandoravisibility.trace``
        takes.
    """
    target = ephemerides[target_id]
    ra0, dec0 = target["ra"][index], target["dec"][index]
    target_jmag = target["vmag"][index] - SOLAR_V_MINUS_J

    xi, eta, flux, jmag, label = [], [], [], [], []
    for body_id in system_ids(target_id):
        body = ephemerides[body_id]
        if body_id == target_id or body["code"][index] in HIDDEN_CODES:
            continue
        center_xi, center_eta = star_field.gnomonic_offsets(
            body["ra"][index], body["dec"][index], ra0, dec0
        )
        dxi, deta, weight = body_samples(body_id, body, index)
        body_jmag = body["vmag"][index] - SOLAR_V_MINUS_J
        xi.append(center_xi + dxi)
        eta.append(center_eta + deta)
        flux.append(10.0 ** (-0.4 * (body_jmag - target_jmag)) * weight)
        jmag.append(np.full(weight.size, body_jmag))
        label.extend([labels[body_id]] + [""] * (weight.size - 1))

    if include_stars:
        stars = star_field.query_stars(SkyCoord(ra0, dec0, unit="deg"), time)
        xi.append(stars["xi"])
        eta.append(stars["eta"])
        flux.append(10.0 ** (-0.4 * (stars["jmag"] - target_jmag)))
        jmag.append(stars["jmag"])
        label.extend(f"J={value:.1f}" for value in stars["jmag"])

    shape_xi, shape_eta, shape_weight = body_samples(target_id, target, index)
    return {
        "xi": np.concatenate(xi) if xi else np.array([]),
        "eta": np.concatenate(eta) if eta else np.array([]),
        "flux_ratio": np.concatenate(flux) if flux else np.array([]),
        "jmag": np.concatenate(jmag) if jmag else np.array([]),
        "label": label,
        "target_shape": {"xi": shape_xi, "eta": shape_eta, "weight": shape_weight},
    }


def system_visible_sources(
    target_id, ephemerides, index, time, labels, radius_arcmin, include_stars=True
):
    """
    VISDA sources around a target at one time step, for ``visda.plot_visible_field``.

    Every body of the target's system that is not hidden (occulted or fully
    eclipsed, see ``HIDDEN_CODES``), the target included, sampled as in
    ``body_samples``, plus Gaia field stars from
    ``star_field.query_visible_field``. Bodies take G = V +
    ``SOLAR_G_MINUS_V`` and BP - RP = ``SOLAR_BP_RP``.

    Parameters
    ----------
    target_id : int
        Horizons id of the body the field is centered on.
    ephemerides : dict
        From ``system_ephemerides``.
    index : int
        Time step.
    time : astropy.time.Time
        That time step, for the stars' proper motion.
    labels : dict
        Horizons id -> display name.
    radius_arcmin : float
        Star cone radius, e.g. from ``visda.field_radius_arcmin``.
    include_stars : bool
        Add Gaia field stars (needs the gaiaoffline database).

    Returns
    -------
    dict
        ``xi``, ``eta``, ``g_mag``, ``bp_mag``, ``bp_rp``, ``weight`` and
        ``label`` per source, in the format ``visda.render_frame`` takes.
        Stars are labeled with their BP when brighter than 12.
    """
    target = ephemerides[target_id]
    ra0, dec0 = target["ra"][index], target["dec"][index]

    xi, eta, g_mag, bp_mag, bp_rp, weight, label = [], [], [], [], [], [], []
    for body_id in system_ids(target_id):
        body = ephemerides[body_id]
        if body_id != target_id and body["code"][index] in HIDDEN_CODES:
            continue
        center_xi, center_eta = star_field.gnomonic_offsets(
            body["ra"][index], body["dec"][index], ra0, dec0
        )
        dxi, deta, body_weight = body_samples(body_id, body, index)
        xi.append(center_xi + dxi)
        eta.append(center_eta + deta)
        g_mag.append(np.full(body_weight.size, body["vmag"][index] + SOLAR_G_MINUS_V))
        bp_mag.append(np.full(body_weight.size, np.nan))
        bp_rp.append(np.full(body_weight.size, SOLAR_BP_RP))
        weight.append(body_weight)
        label.extend([labels[body_id]] + [""] * (body_weight.size - 1))

    if include_stars:
        stars = star_field.query_visible_field(
            SkyCoord(ra0, dec0, unit="deg"), time, radius_arcmin, match_target=False
        )
        xi.append(stars["xi"])
        eta.append(stars["eta"])
        g_mag.append(stars["g_mag"])
        bp_mag.append(stars["bp_mag"])
        bp_rp.append(stars["bp_rp"])
        weight.append(np.ones(stars["xi"].size))
        label.extend(
            f"BP={value:.1f}" if value < 12.0 else "" for value in stars["bp_mag"]
        )

    return {
        "xi": np.concatenate(xi),
        "eta": np.concatenate(eta),
        "g_mag": np.concatenate(g_mag),
        "bp_mag": np.concatenate(bp_mag),
        "bp_rp": np.concatenate(bp_rp),
        "weight": np.concatenate(weight),
        "label": label,
    }


def moon_geometry(moon_id, ephemerides, visibility, times):
    """
    Where a moon is in its orbit, as seen from Pandora.

    Positions are Pandora-centric (see ``pandora_centric``). The orbit
    normal is fit once from the moon's motion about its planet over the
    whole grid, so ``times`` needs at least two steps, less than half an
    orbit apart.

    Parameters
    ----------
    moon_id : int
        Horizons id of the moon.
    ephemerides : dict
        From ``system_ephemerides``; must hold the moon and its planet.
    visibility : pandoravisibility.Visibility
        Supplies the spacecraft orbit.
    times : astropy.time.Time
        The notebook time grid.

    Returns
    -------
    dict of np.ndarray
        host_angle : angle at the planet between the moon and Pandora (deg):
            0 when the moon is straight in front of the planet, 180 straight
            behind it.
        orbital_phase : the moon's orbital phase (deg, 0 to 360) counted
            from in front of the planet (inferior conjunction) in its
            direction of motion: 90 and 270 are the greatest elongations,
            180 is behind the planet.
    """
    spacecraft = _spacecraft_position(visibility, times)
    planet = _geocentric_position(ephemerides[system_ids(moon_id)[0]]) - spacecraft
    moon = _geocentric_position(ephemerides[moon_id]) - spacecraft
    offset = moon - planet  # planet to moon, km
    toward_pandora = -planet / np.linalg.norm(planet, axis=0)
    radial = offset / np.linalg.norm(offset, axis=0)
    host_angle = np.rad2deg(
        np.arccos(np.clip((radial * toward_pandora).sum(axis=0), -1.0, 1.0))
    )

    # One orbit normal from r x dr summed over the grid (the plane barely
    # moves over days), then the phase within that plane from the line of
    # sight's projection onto it
    normal = np.cross(offset[:, :-1], offset[:, 1:], axis=0).sum(axis=1)
    normal = (normal / np.linalg.norm(normal))[:, np.newaxis]
    reference = toward_pandora - (toward_pandora * normal).sum(axis=0) * normal
    reference /= np.linalg.norm(reference, axis=0)
    orbital_phase = np.rad2deg(
        np.arctan2(
            (np.cross(reference, radial, axis=0) * normal).sum(axis=0),
            (reference * radial).sum(axis=0),
        )
    )
    return {"host_angle": host_angle, "orbital_phase": orbital_phase % 360.0}


def specular_point(ephemeris):
    """
    Point on a body where sunlight reflects specularly toward the observer.

    On a sphere this is where the surface normal bisects the directions to
    the Sun and to the observer: the midpoint of the sub-solar and
    sub-observer points. The angle of incidence there is half the phase
    angle. The observer is the Earth's center, which for any body beyond the
    Moon is the same as Pandora to well under 0.01 deg.

    Parameters
    ----------
    ephemeris : dict
        The body's sampled ephemeris (see ``system_ephemerides``).

    Returns
    -------
    latitude, longitude, incidence : np.ndarray
        Specular point latitude and longitude (deg, the body's longitude
        convention) and the angle of incidence there (deg).
    """

    def unit(lat, lon):
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        return np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )

    bisector = unit(ephemeris["sub_lat"], ephemeris["sub_lon"]) + unit(
        ephemeris["sun_lat"], ephemeris["sun_lon"]
    )
    x, y, z = bisector / np.linalg.norm(bisector, axis=0)
    return (
        np.rad2deg(np.arctan2(z, np.hypot(x, y))),
        np.rad2deg(np.arctan2(y, x)) % 360.0,
        0.5 * ephemeris["phase_angle"],
    )


def glint_check(body_id, ephemeris, tolerance_deg):
    """
    Whether the specular point falls on known surface liquid.

    A glint is possible when the specular point (``specular_point``) lies
    within ``tolerance_deg`` of the edge of a feature in
    ``LIQUID_FEATURES``. The tolerance stands in for everything that spreads
    a real glint beyond one point: wave slopes, the Sun's width, scattering
    in the atmosphere, and the rough circular outlines used here.

    Parameters
    ----------
    body_id : int
        Horizons id.
    ephemeris : dict
        The body's sampled ephemeris (see ``system_ephemerides``).
    tolerance_deg : float
        Allowed distance from a feature's edge, in degrees of arc on the
        body.

    Returns
    -------
    possible : np.ndarray of bool
        True where a glint is possible. All False for a body without known
        surface liquid.
    nearest : np.ndarray of str
        Name of the feature whose edge is closest ("" without features).
    distance : np.ndarray
        Arc from the specular point to that feature's edge, deg (NaN without
        features; negative inside it).
    """
    latitude, longitude, _ = specular_point(ephemeris)
    n_time = len(latitude)
    features = LIQUID_FEATURES.get(body_id, [])
    if not features:
        return np.zeros(n_time, bool), np.full(n_time, ""), np.full(n_time, np.nan)
    lat = np.deg2rad(latitude)[:, np.newaxis]
    lon = np.deg2rad(longitude)[:, np.newaxis]
    feature_lat = np.deg2rad([feature[1] for feature in features])
    feature_lon = np.deg2rad([feature[2] for feature in features])
    radius = np.array([feature[3] for feature in features])
    arc = np.rad2deg(
        np.arccos(
            np.clip(
                np.sin(lat) * np.sin(feature_lat)
                + np.cos(lat) * np.cos(feature_lat) * np.cos(lon - feature_lon),
                -1.0,
                1.0,
            )
        )
    )
    beyond_edge = arc - radius
    closest = np.argmin(beyond_edge, axis=1)
    distance = beyond_edge[np.arange(n_time), closest]
    names = np.array([feature[0] for feature in features])
    return distance <= tolerance_deg, names[closest], distance


def fresnel_reflectance(incidence_deg, refractive_index=TITAN_LIQUID_INDEX):
    """
    Fraction of unpolarized light a smooth liquid surface reflects.

    Parameters
    ----------
    incidence_deg : array_like
        Angle of incidence from the surface normal, deg.
    refractive_index : float
        Of the liquid, relative to the gas above it (default
        ``TITAN_LIQUID_INDEX``).

    Returns
    -------
    np.ndarray
        Reflectance, the mean of the s and p polarizations.
    """
    incidence = np.deg2rad(np.asarray(incidence_deg, dtype=float))
    cos_i = np.cos(incidence)
    cos_t = np.sqrt(1.0 - (np.sin(incidence) / refractive_index) ** 2)
    r_s = (cos_i - refractive_index * cos_t) / (cos_i + refractive_index * cos_t)
    r_p = (cos_t - refractive_index * cos_i) / (cos_t + refractive_index * cos_i)
    return 0.5 * (r_s**2 + r_p**2)


def illuminated_disk(body_id, ephemeris):
    """
    How much of a body's disk is sunlit as seen by the observer.

    The lit share of the disk is (1 + cos(phase angle)) / 2, the area
    between the limb and the terminator's ellipse. The disk is the body's
    outline from Horizons' angular diameter (the solid surface: Titan's haze
    reaches higher), flattened by ``POLAR_RATIO`` as seen from the
    sub-observer latitude. Steps where Horizons has the body behind its
    planet or fully in its shadow (``HIDDEN_CODES``) count as unlit. Not
    modeled: partial occultations and eclipses (Horizons ``p``), the planet
    in front of a transiting moon's background, and Saturn's rings.

    Parameters
    ----------
    body_id : int
        Horizons id.
    ephemeris : dict
        The body's sampled ephemeris (see ``system_ephemerides``).

    Returns
    -------
    dict of np.ndarray
        lit_fraction : sunlit share of the disk (0 where hidden).
        lit_area_arcsec2 : sunlit area on the sky, square arcsec.
        lit_area_km2 : the same projected onto the body, square km.
        hidden : True where Horizons has the body occulted or fully eclipsed.
        partial : True during a partial eclipse, whose loss is not modeled.
    """
    semi_major = 0.5 * ephemeris["diameter"]  # arcsec
    sub_lat = np.deg2rad(ephemeris["sub_lat"])
    polar = POLAR_RATIO.get(body_id, 1.0)
    semi_minor = semi_major * np.sqrt(
        (polar * np.cos(sub_lat)) ** 2 + np.sin(sub_lat) ** 2
    )
    hidden = np.isin(ephemeris["code"], HIDDEN_CODES)
    fraction = np.where(
        hidden, 0.0, 0.5 * (1.0 + np.cos(np.deg2rad(ephemeris["phase_angle"])))
    )
    area = fraction * np.pi * semi_major * semi_minor
    km_per_arcsec = ephemeris["delta"] * _AU_KM * np.pi / 648000.0
    return {
        "lit_fraction": fraction,
        "lit_area_arcsec2": area,
        "lit_area_km2": area * km_per_arcsec**2,
        "hidden": hidden,
        "partial": ephemeris["code"] == "p",
    }
