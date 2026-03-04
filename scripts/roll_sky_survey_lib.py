#!/usr/bin/env python
"""
Roll-feasibility sky survey for Pandora — importable library version.

For one Pandora orbit, survey the entire night sky to determine which
coordinates that pass basic visibility constraints (sun > 91°, earth
limb > 20°, moon > 25°) also admit at least one roll angle satisfying
star-tracker keep-out constraints:

    ST sun  ≥ 44°
    ST earth limb ≥ 30°  (same definition as boresight limb angle)
    ST moon ≥ 12°
    st_required = 1  (at least one tracker must pass)

Usage (CLI)
-----------
    python scripts/roll_sky_survey_lib.py

Usage (notebook / import)
-------------------------
    from roll_sky_survey_lib import SurveyConfig, run_survey, print_summary, plot_sky_map

    cfg = SurveyConfig()                       # defaults, or override fields
    results = run_survey(cfg)
    print_summary(results)
    fig = plot_sky_map(results)
"""

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import R_earth as _Re
from astropy.coordinates import GCRS, SkyCoord, get_body
from astropy.time import Time
from matplotlib.colors import BoundaryNorm, ListedColormap

from pandoravisibility import Visibility


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SurveyConfig:
    """All tuneable parameters for a roll sky survey.

    Create with defaults::

        cfg = SurveyConfig()

    Override any subset of parameters::

        cfg = SurveyConfig(max_rolls=1, ra_step=2, t0="2026-06-15T12:00:00")

    Create a modified copy (leaves original unchanged)::

        cfg2 = cfg.replace(roll_step=1, st_sun_min=50)

    View parameters in a notebook::

        cfg          # shows a formatted table
    """

    # TLE
    line1: str = "1 67395U 80229J   26048.99256944  .00000000  00000-0  37770-3 0    09"
    line2: str = "2 67395  97.7994  49.5368 0005873 146.5278 132.7382 14.87786761    09"

    # Epoch
    t0: str = "2026-02-15T18:00:00"

    # Basic boresight constraints (deg)
    sun_min: float = 91.0
    moon_min: float = 25.0
    earth_limb_min: float = 20.0

    # Star-tracker keep-out limits (deg)
    st_sun_min: float = 44.0
    st_earth_min: float = 30.0
    st_moon_min: float = 12.0

    # Per-tracker ST Earth limb overrides (None = use shared st_earth_min)
    st1_earth_min: Optional[float] = None
    st2_earth_min: Optional[float] = None

    # Grid resolution
    ra_step: float = 4.0    # deg
    dec_step: float = 4.0   # deg
    time_step: float = 1.0  # minutes
    roll_step: float = 2.0  # deg

    # Max number of fixed rolls per orbit (1 or 2)
    max_rolls: int = 2

    # Whether to print progress
    verbose: bool = True

    # ── convenience methods ─────────────────────────────────────────

    def replace(self, **kwargs) -> "SurveyConfig":
        """Return a *new* SurveyConfig with the given fields overridden."""
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Return all parameters as a plain dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SurveyConfig":
        """Create a SurveyConfig from a dict (ignores unknown keys)."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def __repr__(self) -> str:
        defaults = SurveyConfig()
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            default_val = getattr(defaults, f.name)
            marker = " *" if val != default_val else ""
            parts.append(f"  {f.name} = {val!r}{marker}")
        body = "\n".join(parts)
        return f"SurveyConfig(\n{body}\n)  # * = changed from default"

    def _repr_html_(self) -> str:
        """Rich display for Jupyter notebooks."""
        defaults = SurveyConfig()
        _groups = [
            ("TLE", ["line1", "line2"]),
            ("Epoch", ["t0"]),
            ("Boresight constraints (deg)", ["sun_min", "moon_min", "earth_limb_min"]),
            ("ST keep-out limits (deg)", ["st_sun_min", "st_earth_min", "st_moon_min",
                                          "st1_earth_min", "st2_earth_min"]),
            ("Grid resolution", ["ra_step", "dec_step", "time_step", "roll_step"]),
            ("Roll search", ["max_rolls"]),
            ("Misc", ["verbose"]),
        ]
        rows = []
        for group, names in _groups:
            rows.append(
                f'<tr><td colspan="3" style="background:#eee;font-weight:bold;'
                f'padding:4px 8px">{group}</td></tr>'
            )
            for name in names:
                val = getattr(self, name)
                default_val = getattr(defaults, name)
                changed = val != default_val
                if changed:
                    td_style = "padding:2px 8px;color:#d32f2f;font-weight:bold"
                else:
                    td_style = "padding:2px 8px"
                marker = " &#9998;" if changed else ""
                rows.append(
                    f"<tr><td style='padding:2px 8px'><code>{name}</code></td>"
                    f"<td style='{td_style}'>{val!r}{marker}</td>"
                    f"<td style='padding:2px 8px;color:#888'>default: {default_val!r}</td></tr>"
                )
        table = "\n".join(rows)
        return (
            '<table style="border-collapse:collapse;font-size:13px">'
            f"{table}</table>"
        )


# ═══════════════════════════════════════════════════════════════════
# Results container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SurveyResults:
    """All outputs from a roll sky survey run."""

    config: SurveyConfig

    # Grid geometry
    ra_vals: np.ndarray       # (N_ra,)
    dec_vals: np.ndarray      # (N_dec,)
    ra_grid: np.ndarray       # (N_dec, N_ra)
    dec_grid: np.ndarray      # (N_dec, N_ra)
    ra_flat: np.ndarray       # (N_sky,)
    dec_flat: np.ndarray      # (N_sky,)

    # Orbital info
    period_min: float
    times: object             # astropy Time array
    n_time: int

    # Per-target (flattened sky grid) results
    n_visible: np.ndarray             # (N_sky,) visible timesteps per target
    has_fixed_roll: np.ndarray        # (N_sky,) bool
    n_rolls_used: np.ndarray          # (N_sky,) 0, 1, or 2
    best_roll_deg: np.ndarray         # (N_sky,)
    second_roll_deg: np.ndarray       # (N_sky,)
    best_roll_frac: np.ndarray        # (N_sky,)
    sun_roll_deg: np.ndarray          # (N_sky,)
    roll_offset_deg: np.ndarray       # (N_sky,)
    worst_solar_incidence_deg: np.ndarray  # (N_sky,)
    worst_solar_power_frac: np.ndarray     # (N_sky,)

    # Precomputed ephemeris (kept for plotting)
    sun_unit: np.ndarray      # (3, N_t)


# ═══════════════════════════════════════════════════════════════════
# Vectorised helpers
# ═══════════════════════════════════════════════════════════════════

def fast_sep_deg(a, b):
    """Angular separation in degrees.  a: (3,) or (3,N); b: (3,N)."""
    dot = np.sum(a * b, axis=0)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def fast_limb_deg(target_unit, zenith_u, limb_rad):
    """Earth limb angle: elevation above horizon + angular radius of Earth."""
    dot = np.sum(target_unit * zenith_u, axis=0)
    elev = np.arcsin(np.clip(dot, -1.0, 1.0))
    return np.rad2deg(elev + limb_rad)


# ═══════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════

def run_survey(config: Optional[SurveyConfig] = None) -> SurveyResults:
    """
    Run a full roll-feasibility sky survey for one Pandora orbit.

    Parameters
    ----------
    config : SurveyConfig, optional
        Survey parameters.  Uses ``SurveyConfig()`` defaults if omitted.

    Returns
    -------
    SurveyResults
        All computed arrays, ready for ``print_summary`` / ``plot_sky_map``.
    """
    if config is None:
        config = SurveyConfig()
    cfg = config
    log = print if cfg.verbose else (lambda *a, **k: None)

    # ── Setup ───────────────────────────────────────────────────────
    vis = Visibility(cfg.line1, cfg.line2)
    period = vis.get_period().to(u.min).value
    log(f"Orbital period: {period:.2f} min")

    t0 = Time(cfg.t0) if isinstance(cfg.t0, str) else cfg.t0
    n_time = int(np.ceil(period / cfg.time_step)) + 1
    times = t0 + np.linspace(0, period, n_time) * u.min
    N_t = len(times)
    log(f"Time steps: {N_t}")

    # Sky grid
    ra_vals = np.arange(0, 360, cfg.ra_step)
    dec_vals = np.arange(-90, 90 + cfg.dec_step / 2, cfg.dec_step)
    dec_vals = np.clip(dec_vals, -90, 90)  # guard against overshoot for non-divisor steps
    RA_grid, DEC_grid = np.meshgrid(ra_vals, dec_vals)
    ra_flat = RA_grid.ravel()
    dec_flat = DEC_grid.ravel()
    N_sky = len(ra_flat)
    log(f"Sky grid: {len(ra_vals)} × {len(dec_vals)} = {N_sky} points")

    # Roll angles
    roll_degs = np.arange(0, 360, cfg.roll_step)
    N_roll = len(roll_degs)
    log(f"Roll sweep: {N_roll} angles ({cfg.roll_step}° steps)")

    # ST body vectors
    st1_body = np.array(Visibility._get_star_tracker_body_xyz(1))
    st2_body = np.array(Visibility._get_star_tracker_body_xyz(2))

    # Per-tracker Earth limb limits (resolve None → shared value)
    st1_earth_limit = cfg.st1_earth_min if cfg.st1_earth_min is not None else cfg.st_earth_min
    st2_earth_limit = cfg.st2_earth_min if cfg.st2_earth_min is not None else cfg.st_earth_min

    # ── Precompute ephemeris ────────────────────────────────────────
    log("Precomputing ephemeris for all time steps...")
    observer_location = vis._get_observer_location(times)
    obs_gcrs = observer_location.get_gcrs(obstime=times)

    obs_xyz = obs_gcrs.cartesian.xyz.to(u.m).value   # (3, N_t)
    obs_dist = np.linalg.norm(obs_xyz, axis=0)        # (N_t,)
    zenith_unit = obs_xyz / obs_dist[np.newaxis, :]    # (3, N_t)

    R_EARTH_M = _Re.to(u.m).value
    limb_angle_rad = np.arccos(R_EARTH_M / obs_dist)   # (N_t,)

    sun_body_coord = get_body("sun", time=times, location=observer_location)
    sun_xyz = sun_body_coord.cartesian.xyz.value
    sun_unit = sun_xyz / np.linalg.norm(sun_xyz, axis=0, keepdims=True)

    moon_body_coord = get_body("moon", time=times, location=observer_location)
    moon_xyz = moon_body_coord.cartesian.xyz.value
    moon_unit = moon_xyz / np.linalg.norm(moon_xyz, axis=0, keepdims=True)
    log("Ephemeris done.")

    # ── Target directions in GCRS ───────────────────────────────────
    log("Computing target directions...")
    ref_time = times[0]
    targets = SkyCoord(ra=ra_flat * u.deg, dec=dec_flat * u.deg, frame="icrs")
    targets_gcrs = targets.transform_to(GCRS(obstime=ref_time))
    tgt_xyz = targets_gcrs.cartesian.xyz.value
    tgt_unit = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)

    # ── Basic boresight constraints ─────────────────────────────────
    log("Evaluating basic boresight constraints...")

    # All-pairs dot products: each target (col of tgt_unit) dotted with
    # each timestep (col of body_unit).  tgt_unit is (3, N_sky), so
    # tgt_unit.T @ body_unit gives (N_sky, N_t) dot-product matrix.
    sun_dots = tgt_unit.T @ sun_unit          # (N_sky, N_t)
    sun_sep = np.rad2deg(np.arccos(np.clip(sun_dots, -1.0, 1.0)))

    moon_dots = tgt_unit.T @ moon_unit        # (N_sky, N_t)
    moon_sep = np.rad2deg(np.arccos(np.clip(moon_dots, -1.0, 1.0)))

    zenith_dots = tgt_unit.T @ zenith_unit    # (N_sky, N_t)
    boresight_elev = np.arcsin(np.clip(zenith_dots, -1.0, 1.0))
    boresight_limb = np.rad2deg(boresight_elev + limb_angle_rad[np.newaxis, :])

    basic_mask = (
        (sun_sep >= cfg.sun_min)
        & (moon_sep >= cfg.moon_min)
        & (boresight_limb >= cfg.earth_limb_min)
    )

    n_visible = basic_mask.sum(axis=1)
    log(f"  {(n_visible > 0).sum()} / {N_sky} coordinates visible at ≥1 timestep")

    # ── Roll sweep ──────────────────────────────────────────────────
    log(f"Sweeping roll angles for star-tracker constraints (max_rolls={cfg.max_rolls})...")
    log(f"  (requiring {'a single fixed' if cfg.max_rolls == 1 else 'up to two fixed'}"
        f" roll that cover ALL visible timesteps)")

    # Result arrays
    has_fixed_roll = np.zeros(N_sky, dtype=bool)
    n_rolls_used = np.zeros(N_sky, dtype=int)
    best_roll_deg = np.full(N_sky, np.nan)
    second_roll_deg = np.full(N_sky, np.nan)
    best_roll_frac = np.zeros(N_sky)
    sun_roll_deg = np.full(N_sky, np.nan)
    roll_offset_deg = np.full(N_sky, np.nan)
    worst_solar_incidence_deg = np.full(N_sky, np.nan)
    worst_solar_power_frac = np.full(N_sky, np.nan)

    visible_idx = np.where(n_visible > 0)[0]
    log(f"  Processing {len(visible_idx)} visible coordinates...")

    for count, i in enumerate(visible_idx):
        if cfg.verbose and count % 500 == 0 and count > 0:
            print(f"    {count}/{len(visible_idx)}...")

        z = tgt_unit[:, i]
        t_mask = basic_mask[i]
        t_idx = np.where(t_mask)[0]
        N_vis = len(t_idx)
        if N_vis == 0:
            continue

        sun_vis = sun_unit[:, t_idx]
        moon_vis = moon_unit[:, t_idx]
        zenith_vis = zenith_unit[:, t_idx]
        limb_vis = limb_angle_rad[t_idx]

        # Sun-constrained roll angle for this target
        mid_sun = sun_vis[:, N_vis // 2]
        y_sun = np.cross(mid_sun, z)
        y_sun_norm = np.linalg.norm(y_sun)
        if y_sun_norm > 1e-10:
            y_sun = y_sun / y_sun_norm
            x_sun = np.cross(y_sun, z)
            x_sun = x_sun / np.linalg.norm(x_sun)
            north = np.array([0.0, 0.0, 1.0])
            north_proj = north - np.dot(north, z) * z
            nn = np.linalg.norm(north_proj)
            if nn < 1e-8:
                east = np.array([1.0, 0.0, 0.0])
                north_proj = east - np.dot(east, z) * z
                nn = np.linalg.norm(north_proj)
            x_ref = north_proj / nn
            y_ref = np.cross(z, x_ref)
            y_ref = y_ref / np.linalg.norm(y_ref)
            sun_roll_deg[i] = np.degrees(np.arctan2(
                np.dot(y_ref, x_sun), np.dot(x_ref, x_sun)
            ))

        # Build coverage matrix: coverage[r, t] = True if roll r works at time t
        coverage = np.zeros((N_roll, N_vis), dtype=bool)

        for r, roll_d in enumerate(roll_degs):
            roll_rad = np.deg2rad(roll_d)
            x_pay, y_pay = Visibility._roll_attitude(z, roll_rad)

            st1_eci = x_pay * st1_body[0] + y_pay * st1_body[1] + z * st1_body[2]
            st1_eci = st1_eci / np.linalg.norm(st1_eci)
            st2_eci = x_pay * st2_body[0] + y_pay * st2_body[1] + z * st2_body[2]
            st2_eci = st2_eci / np.linalg.norm(st2_eci)

            st1_sun = fast_sep_deg(st1_eci[:, np.newaxis], sun_vis)
            st1_moon = fast_sep_deg(st1_eci[:, np.newaxis], moon_vis)
            st1_limb = fast_limb_deg(st1_eci[:, np.newaxis], zenith_vis, limb_vis)
            st1_ok = ((st1_sun >= cfg.st_sun_min)
                      & (st1_moon >= cfg.st_moon_min)
                      & (st1_limb >= st1_earth_limit))

            st2_sun = fast_sep_deg(st2_eci[:, np.newaxis], sun_vis)
            st2_moon = fast_sep_deg(st2_eci[:, np.newaxis], moon_vis)
            st2_limb = fast_limb_deg(st2_eci[:, np.newaxis], zenith_vis, limb_vis)
            st2_ok = ((st2_sun >= cfg.st_sun_min)
                      & (st2_moon >= cfg.st_moon_min)
                      & (st2_limb >= st2_earth_limit))

            coverage[r] = st1_ok | st2_ok

        # --- Check single roll ---
        roll_counts = coverage.sum(axis=1)
        best_r_idx = np.argmax(roll_counts)
        best_frac = roll_counts[best_r_idx] / N_vis
        best_roll_frac[i] = best_frac
        best_roll_deg[i] = roll_degs[best_r_idx]

        if roll_counts[best_r_idx] == N_vis:
            has_fixed_roll[i] = True
            n_rolls_used[i] = 1
        elif cfg.max_rolls >= 2:
            # --- Check pairs ---
            found_pair = False
            useful = np.where(roll_counts > 0)[0]
            useful = useful[np.argsort(-roll_counts[useful])]
            for j, r1 in enumerate(useful):
                gaps = ~coverage[r1]
                n_gaps = gaps.sum()
                if n_gaps == 0:
                    has_fixed_roll[i] = True
                    n_rolls_used[i] = 1
                    best_roll_deg[i] = roll_degs[r1]
                    best_roll_frac[i] = 1.0
                    found_pair = True
                    break
                gap_coverage = coverage[useful, :][:, gaps].sum(axis=1)
                fills = np.where(gap_coverage == n_gaps)[0]
                if len(fills) > 0:
                    r2 = useful[fills[np.argmax(roll_counts[useful[fills]])]]
                    has_fixed_roll[i] = True
                    n_rolls_used[i] = 2
                    best_roll_deg[i] = roll_degs[r1]
                    second_roll_deg[i] = roll_degs[r2]
                    best_roll_frac[i] = 1.0
                    found_pair = True
                    break
            if not found_pair:
                best_roll_deg[i] = roll_degs[best_r_idx]
                best_roll_frac[i] = best_frac

        # Signed offset from Sun-constrained roll
        if not np.isnan(best_roll_deg[i]) and not np.isnan(sun_roll_deg[i]):
            diff = best_roll_deg[i] - sun_roll_deg[i]
            diff = (diff + 180) % 360 - 180
            roll_offset_deg[i] = diff

        # Solar array incidence angle
        rolls_to_check = []
        if not np.isnan(best_roll_deg[i]):
            rolls_to_check.append(best_roll_deg[i])
        if not np.isnan(second_roll_deg[i]):
            rolls_to_check.append(second_roll_deg[i])
        if rolls_to_check:
            worst_inc = 0.0
            for rd in rolls_to_check:
                roll_rad = np.deg2rad(rd)
                x_pay, y_pay = Visibility._roll_attitude(z, roll_rad)
                cos_sy = np.sum(y_pay[:, np.newaxis] * sun_vis, axis=0)
                cos_sy = np.clip(cos_sy, -1.0, 1.0)
                theta_sy = np.rad2deg(np.arccos(np.abs(cos_sy)))
                incidence = 90.0 - theta_sy
                worst_inc = max(worst_inc, incidence.max())
            worst_solar_incidence_deg[i] = worst_inc
            worst_solar_power_frac[i] = np.cos(np.deg2rad(worst_inc))

    log("Roll sweep complete.")

    # Normalize roll angles to [-180, 180]
    best_roll_deg = (best_roll_deg + 180) % 360 - 180
    second_roll_deg = (second_roll_deg + 180) % 360 - 180
    sun_roll_deg = (sun_roll_deg + 180) % 360 - 180

    return SurveyResults(
        config=cfg,
        ra_vals=ra_vals,
        dec_vals=dec_vals,
        ra_grid=RA_grid,
        dec_grid=DEC_grid,
        ra_flat=ra_flat,
        dec_flat=dec_flat,
        period_min=period,
        times=times,
        n_time=N_t,
        n_visible=n_visible,
        has_fixed_roll=has_fixed_roll,
        n_rolls_used=n_rolls_used,
        best_roll_deg=best_roll_deg,
        second_roll_deg=second_roll_deg,
        best_roll_frac=best_roll_frac,
        sun_roll_deg=sun_roll_deg,
        roll_offset_deg=roll_offset_deg,
        worst_solar_incidence_deg=worst_solar_incidence_deg,
        worst_solar_power_frac=worst_solar_power_frac,
        sun_unit=sun_unit,
    )


# ═══════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════

def print_summary(results: SurveyResults):
    """Print a text summary of survey results."""
    r = results
    cfg = r.config
    N_sky = len(r.ra_flat)

    n_night = (r.n_visible > 0).sum()
    n_solved = r.has_fixed_roll.sum()
    n_1roll = (r.has_fixed_roll & (r.n_rolls_used == 1)).sum()
    n_2roll = (r.has_fixed_roll & (r.n_rolls_used == 2)).sum()
    n_partial = ((r.best_roll_frac > 0) & ~r.has_fixed_roll).sum()
    n_none = n_night - n_solved - n_partial

    print(f"\n{'='*60}")
    print(f"Sky survey results (one orbit, {r.period_min:.1f} min, max_rolls={cfg.max_rolls})")
    print(f"{'='*60}")
    print(f"  Sky grid points:                  {N_sky}")
    print(f"  Night sky (sun > {cfg.sun_min}°):           {n_night}  ({100*n_night/N_sky:.1f}%)")
    print(f"  Solved (roll(s) cover ALL vis):   {n_solved}  ({100*n_solved/max(n_night,1):.1f}% of night)")
    if cfg.max_rolls >= 2:
        print(f"    └ with 1 roll:                  {n_1roll}")
        print(f"    └ with 2 rolls:                 {n_2roll}")
    print(f"  Best roll(s) cover only PARTIAL:  {n_partial}  ({100*n_partial/max(n_night,1):.1f}% of night)")
    print(f"  No roll works at all:             {n_none}  ({100*n_none/max(n_night,1):.1f}% of night)")
    print(f"{'='*60}")

    night_mask = r.n_visible > 0
    valid_offsets = r.roll_offset_deg[night_mask & ~np.isnan(r.roll_offset_deg)]
    if len(valid_offsets) > 0:
        abs_off = np.abs(valid_offsets)
        print(f"  Roll offset from Sun×Z attitude:")
        print(f"    median |offset| = {np.median(abs_off):.1f}°, "
              f"mean = {np.mean(abs_off):.1f}°, max = {abs_off.max():.1f}°")
        for thresh in [10, 20, 30, 45, 90]:
            pct = 100 * (abs_off <= thresh).sum() / len(abs_off)
            print(f"    |offset| ≤ {thresh:>2}°: {pct:5.1f}%")

    sa_valid = r.worst_solar_power_frac[night_mask & ~np.isnan(r.worst_solar_power_frac)]
    sa_inc = r.worst_solar_incidence_deg[night_mask & ~np.isnan(r.worst_solar_incidence_deg)]
    if len(sa_valid) > 0:
        print(f"  Solar array (worst-case over visible orbit):")
        print(f"    Incidence angle:  median {np.median(sa_inc):.1f}°, "
              f"mean {np.mean(sa_inc):.1f}°, max {sa_inc.max():.1f}°")
        print(f"    Power fraction:   median {np.median(sa_valid)*100:.1f}%, "
              f"mean {np.mean(sa_valid)*100:.1f}%, min {sa_valid.min()*100:.1f}%")
        fixed_mask = r.has_fixed_roll & ~np.isnan(r.worst_solar_power_frac)
        if fixed_mask.any():
            sa_fixed = r.worst_solar_power_frac[fixed_mask]
            sa_inc_fixed = r.worst_solar_incidence_deg[fixed_mask]
            print(f"    (fixed-roll targets only):")
            print(f"      Incidence:  median {np.median(sa_inc_fixed):.1f}°, "
                  f"mean {np.mean(sa_inc_fixed):.1f}°, max {sa_inc_fixed.max():.1f}°")
            print(f"      Power frac: median {np.median(sa_fixed)*100:.1f}%, "
                  f"mean {np.mean(sa_fixed)*100:.1f}%, min {sa_fixed.min()*100:.1f}%")
            for pthresh in [90, 80, 70, 50]:
                pct = 100 * (sa_fixed >= pthresh / 100).sum() / len(sa_fixed)
                print(f"      power ≥ {pthresh}%: {pct:5.1f}% of targets")

    n_partial = ((r.best_roll_frac > 0) & ~r.has_fixed_roll).sum()
    if n_partial > 0:
        partial_mask = (r.best_roll_frac > 0) & ~r.has_fixed_roll & (r.n_visible > 0)
        partial_fracs = r.best_roll_frac[partial_mask]
        print(f"  Partial coverage: median {np.median(partial_fracs)*100:.0f}%, "
              f"min {partial_fracs.min()*100:.0f}%, max {partial_fracs.max()*100:.0f}%")


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_sky_map(results: SurveyResults, savepath: Optional[str] = None,
                 show: bool = True):
    """
    Generate a 5-panel antisolar-centred Mollweide sky map.

    Parameters
    ----------
    results : SurveyResults
    savepath : str, optional
        If given, save figure to this path.
    show : bool
        Whether to call ``plt.show()``.  Set False in notebooks
        that display figures inline.

    Returns
    -------
    matplotlib.figure.Figure
    """
    r = results
    cfg = r.config
    N_sky = len(r.ra_flat)
    N_t = r.n_time

    # Per-tracker Earth limb limits (resolve None → shared value)
    st1_earth_limit = cfg.st1_earth_min if cfg.st1_earth_min is not None else cfg.st_earth_min
    st2_earth_limit = cfg.st2_earth_min if cfg.st2_earth_min is not None else cfg.st_earth_min

    print("Generating sky map...")

    # ── Derived quantities ───────────────────────────────────────────
    # Actual time spacing (may differ slightly from cfg.time_step due to linspace)
    actual_step_min = r.period_min / (N_t - 1) if N_t > 1 else cfg.time_step
    # Covered duration in minutes (best roll(s))
    covered_min = r.best_roll_frac * r.n_visible * actual_step_min
    # Total visible duration in minutes
    visible_min = r.n_visible * actual_step_min

    # ── Categorise each pixel ───────────────────────────────────────
    category = np.zeros(N_sky, dtype=int)
    category[r.n_visible > 0] = 1
    category[(r.best_roll_frac > 0) & ~r.has_fixed_roll] = 2
    category[r.has_fixed_roll & (r.n_rolls_used == 1)] = 3
    category[r.has_fixed_roll & (r.n_rolls_used == 2)] = 4

    CAT = category.reshape(r.ra_grid.shape)
    FRAC = r.best_roll_frac.reshape(r.ra_grid.shape)
    OFFSET = r.roll_offset_deg.reshape(r.ra_grid.shape)
    SOLAR = r.worst_solar_power_frac.reshape(r.ra_grid.shape)
    DURATION = covered_min.reshape(r.ra_grid.shape)

    # ── Antisolar centering ─────────────────────────────────────────
    mid_sun_eci = r.sun_unit[:, N_t // 2]
    sun_ra_rad = np.arctan2(mid_sun_eci[1], mid_sun_eci[0])
    antisolar_ra_rad = sun_ra_rad + np.pi
    antisolar_ra_deg = np.degrees(antisolar_ra_rad) % 360
    print(f"  Sun RA = {np.degrees(sun_ra_rad):.1f}°, "
          f"antisolar RA = {antisolar_ra_deg:.1f}°")

    ra_plot = np.deg2rad(r.ra_vals) - antisolar_ra_rad
    ra_plot = (ra_plot + np.pi) % (2 * np.pi) - np.pi
    dec_plot = np.deg2rad(r.dec_vals)

    sort_idx = np.argsort(ra_plot)
    ra_plot = ra_plot[sort_idx]
    CAT_sorted = CAT[:, sort_idx]
    FRAC_sorted = FRAC[:, sort_idx]
    OFFSET_sorted = OFFSET[:, sort_idx]
    SOLAR_sorted = SOLAR[:, sort_idx]
    DURATION_sorted = DURATION[:, sort_idx]

    RA_P, DEC_P = np.meshgrid(ra_plot, dec_plot)

    # ── Build figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(14, 25),
                             subplot_kw={"projection": "mollweide"})

    def label_axes(ax):
        ax.plot(0, 0, marker="*", color="gold", markersize=12, zorder=5,
                markeredgecolor="k", markeredgewidth=0.5)
        ax.text(0.01, 0.02, "antisolar", fontsize=8, color="gold",
                transform=ax.transAxes)
        tick_lons_deg = np.arange(-150, 180, 30)
        tick_lons_rad = np.deg2rad(tick_lons_deg)
        true_ra = (tick_lons_deg + antisolar_ra_deg) % 360
        labels = [f"{int(rv)}°" for rv in true_ra]
        ax.set_xticks(tick_lons_rad)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, alpha=0.3)

    # Panel 1: Category map
    ax = axes[0]
    if cfg.max_rolls >= 2:
        cmap = ListedColormap(["#cccccc", "#d32f2f", "#fdd835", "#388e3c", "#1976d2"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
        tick_vals = [0, 1, 2, 3, 4]
        tick_labels = ["Day side", "No roll", "Partial", "1 roll OK", "2 rolls OK"]
    else:
        cmap = ListedColormap(["#cccccc", "#d32f2f", "#fdd835", "#388e3c"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        tick_vals = [0, 1, 2, 3]
        tick_labels = ["Day side", "No roll", "Partial", "Fixed roll OK"]
    im = ax.pcolormesh(RA_P, DEC_P, CAT_sorted, cmap=cmap, norm=norm, shading="auto")
    cbar = fig.colorbar(im, ax=ax, ticks=tick_vals, shrink=0.7, pad=0.08)
    cbar.ax.set_yticklabels(tick_labels)
    # Build Earth limb label (show per-tracker if different)
    if st1_earth_limit == st2_earth_limit:
        limb_label = f"Earth limb ≥ {st1_earth_limit}°"
    else:
        limb_label = f"ST1 limb ≥ {st1_earth_limit}°, ST2 limb ≥ {st2_earth_limit}°"
    ax.set_title(
        f"Roll feasibility (max_rolls={cfg.max_rolls}) — one orbit ({r.period_min:.0f} min)\n"
        f"ST keep-outs: Sun ≥ {cfg.st_sun_min}°, {limb_label}, "
        f"Moon ≥ {cfg.st_moon_min}°",
        fontsize=12,
    )
    label_axes(ax)

    # Panel 2: Coverage fraction
    ax2 = axes[1]
    frac_masked = np.ma.masked_where(CAT_sorted == 0, FRAC_sorted)
    im2 = ax2.pcolormesh(RA_P, DEC_P, frac_masked, cmap="RdYlGn", vmin=0, vmax=1,
                          shading="auto")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.7, pad=0.08)
    cbar2.set_label("Best roll(s) time coverage")
    ax2.set_title("Best roll(s): fraction of visible orbit time with ST constraints met",
                  fontsize=12)
    label_axes(ax2)

    # Panel 3: Roll offset from Sun×Z
    ax3 = axes[2]
    offset_masked = np.ma.masked_where(CAT_sorted == 0, OFFSET_sorted)
    im3 = ax3.pcolormesh(RA_P, DEC_P, offset_masked, cmap="RdBu_r", vmin=-180, vmax=180,
                          shading="auto")
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.7, pad=0.08)
    cbar3.set_label("Roll offset from Sun×Z (deg)")
    ax3.set_title("Best fixed roll − Sun-constrained roll (positive = rotated from Sun×Z)",
                  fontsize=12)
    label_axes(ax3)

    # Panel 4: Solar array power
    ax4 = axes[3]
    solar_masked = np.ma.masked_where(CAT_sorted == 0, SOLAR_sorted)
    im4 = ax4.pcolormesh(RA_P, DEC_P, solar_masked * 100, cmap="RdYlGn", vmin=0, vmax=100,
                          shading="auto")
    cbar4 = fig.colorbar(im4, ax=ax4, shrink=0.7, pad=0.08)
    cbar4.set_label("Power fraction (%)")
    ax4.set_title(
        "Worst-case solar array power fraction at best fixed roll\n"
        "(arrays rotate around Y; Sun×Z default = 100%)",
        fontsize=12,
    )
    label_axes(ax4)

    # Panel 5: Covered duration in minutes
    ax5 = axes[4]
    dur_masked = np.ma.masked_where(CAT_sorted == 0, DURATION_sorted)
    max_dur = visible_min.max()
    im5 = ax5.pcolormesh(RA_P, DEC_P, dur_masked, cmap="viridis",
                          vmin=0, vmax=max_dur, shading="auto")
    cbar5 = fig.colorbar(im5, ax=ax5, shrink=0.7, pad=0.08)
    cbar5.set_label("Duration (min)")
    ax5.set_title(
        "Visibility duration covered by best roll(s) (minutes)\n"
        f"Max possible = {max_dur:.0f} min (full visible arc)",
        fontsize=12,
    )
    label_axes(ax5)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    if show:
        plt.show()

    return fig


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_survey()
    print_summary(results)
    plot_sky_map(results, savepath="scripts/roll_sky_survey.png")
