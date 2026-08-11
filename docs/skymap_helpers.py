"""
Helper functions for ``sky_duty_cycle.ipynb``.

The notebook sweeps a regular RA/Dec grid over a time window and asks, for
every grid point, what fraction of that window Pandora can observe it — the
duty cycle.  This module holds the pieces:

* time grid and sky grid construction;
* :func:`duty_cycle_map`, the sweep itself, which calls
  ``Visibility.get_visibility_best_roll`` once per grid point and collects
  the duty cycle, the longest continuous window, the window count and the
  roll angle the search settled on;
* the maps and the area-weighted summary statistics.

Nothing here is Pandora-specific beyond the ``Visibility`` instance handed
in, so the same functions work for any keep-out configuration.
"""

import time as _clock

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time

__all__ = [
    "resolve_start_time",
    "build_time_grid",
    "build_sky_grid",
    "duty_cycle_map",
    "body_radec",
    "plot_duty_rectangular",
    "plot_roll_map",
    "roll_summary",
    "sky_area_weights",
    "duty_summary",
    "plot_duty_distribution",
]

#: Colour ramp for every duty-cycle panel.  Perceptually uniform and
#: monotonic in lightness, so the printed and colour-blind versions still
#: order correctly.
DUTY_CMAP = "viridis"

#: Colour ramp for the longest-window panel, kept distinct from DUTY_CMAP
#: so the two panels are not mistaken for the same quantity.
WINDOW_CMAP = "magma"

#: Colour ramp for the roll angle.  Cyclic, because -180 and +180 deg are
#: the same attitude and any ramp with two different ends would print
#: them as opposites.
ROLL_CMAP = "twilight"

#: Colour ramp for the roll spread — an ordinary magnitude, so an
#: ordinary sequential ramp, chosen to read differently from DUTY_CMAP.
SPREAD_CMAP = "cividis"

#: Fill for grid points with no value: never visible, or no roll angle.
NO_DATA_COLOR = "0.85"


# ----------------------------------------------------------------------
# Time and sky grids
# ----------------------------------------------------------------------

def resolve_start_time(start_utc=None):
    """
    Turn the notebook's ``start_utc`` setting into a :class:`~astropy.time.Time`.

    Parameters
    ----------
    start_utc : str or None
        An ISO UTC timestamp such as ``"2026-08-10T18:30:00"``, or None to
        use today's date at 00:00 UTC.

    Returns
    -------
    astropy.time.Time
        The window start, in UTC.
    """
    if start_utc is None:
        return Time(f"{Time.now().utc.iso[:10]}T00:00:00", scale="utc")
    return Time(start_utc, scale="utc")


def build_time_grid(start, duration_days, step_minutes):
    """
    Regular UTC time grid covering ``[start, start + duration_days)``.

    Parameters
    ----------
    start : astropy.time.Time
        Window start.
    duration_days : float
        Window length in days.  This is the "X" the duty cycle is measured
        over, so every returned fraction is a fraction of this span.
    step_minutes : float
        Spacing between samples, in minutes.  The duty cycle is the
        fraction of samples that are visible, so this also sets how finely
        the visibility windows are resolved — 10 min is a reasonable
        compromise against a ~97 min orbit.

    Returns
    -------
    astropy.time.Time
        Array of observation times.
    """
    if duration_days <= 0:
        raise ValueError("duration_days must be positive")
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    n_steps = int(round(duration_days * 24.0 * 60.0 / step_minutes))
    if n_steps < 2:
        raise ValueError("the window holds fewer than two time steps")
    return start + np.arange(n_steps) * step_minutes * u.min


def build_sky_grid(ra_step, dec_step):
    """
    Regular RA/Dec grid covering the whole sky.

    RA runs from 0 up to but not including 360 — the seam is closed by the
    plotting helpers rather than duplicated here.  Dec runs from -90 to
    +90 inclusive, so both poles are sampled.

    Parameters
    ----------
    ra_step, dec_step : float
        Grid spacing in degrees.

    Returns
    -------
    ra_vals : ndarray, shape (n_ra,)
        Right ascension values, degrees.
    dec_vals : ndarray, shape (n_dec,)
        Declination values, degrees.
    """
    ra_vals = np.arange(0.0, 360.0, ra_step)
    n_dec = int(np.floor(180.0 / dec_step)) + 1
    dec_vals = np.clip(-90.0 + np.arange(n_dec) * dec_step, -90.0, 90.0)
    if dec_vals[-1] < 90.0:
        dec_vals = np.append(dec_vals, 90.0)
    return ra_vals, dec_vals


# ----------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------

def _run_lengths(mask):
    """
    Lengths of the contiguous True runs in a boolean array.

    Pure index arithmetic on a uniform grid, which is all the notebook
    needs and far cheaper than slicing an astropy ``Time`` per run.

    Parameters
    ----------
    mask : ndarray of bool

    Returns
    -------
    ndarray of int
        One entry per run, in order.  Empty when *mask* is all False.
    """
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return stops - starts


def _format_duration(seconds):
    """``"1m 12s"``-style duration, for the progress line."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def duty_cycle_map(visibility, ra_vals, dec_vals, times,
                   roll_step=2 * u.deg, orbit_time_step=1 * u.min,
                   progress=True):
    """
    Duty cycle for every point of a sky grid over one time window.

    Each grid point is evaluated with
    ``Visibility.get_visibility_best_roll``, which picks the best fixed
    roll angle for each orbit before testing the star tracker keep-outs.
    The orbit grouping and the Sun/Moon ephemeris do not depend on the
    target, so the ``Visibility`` instance caches them after the first
    grid point and every later point reuses them.

    Parameters
    ----------
    visibility : pandoravisibility.Visibility
        Configured with the keep-outs to apply.
    ra_vals, dec_vals : ndarray
        Grid axes in degrees, from :func:`build_sky_grid`.
    times : astropy.time.Time
        Observation times, from :func:`build_time_grid`.
    roll_step : astropy.units.Quantity
        Roll sweep resolution.  Coarser is faster and costs a little
        accuracy near the edges of the star tracker keep-outs.
    orbit_time_step : astropy.units.Quantity
        Sampling interval inside each orbit for the roll search.
    progress : bool
        Print a progress line with an elapsed/remaining estimate.

    Returns
    -------
    dict
        ra_vals, dec_vals : the grid axes, echoed back.
        duty : ndarray, shape (n_dec, n_ra)
            Percentage of the window that is observable, all constraints
            applied.
        boresight_duty : ndarray, shape (n_dec, n_ra)
            The same, with the boresight keep-outs only — the star
            trackers and the roll search are ignored.  The gap between the
            two is what the tracker constraints cost.
        longest_window : ndarray, shape (n_dec, n_ra)
            Longest continuous visible stretch, in minutes.
        n_windows : ndarray, shape (n_dec, n_ra)
            Number of separate visible stretches.
        mean_roll : ndarray, shape (n_dec, n_ra)
            Circular mean of the orbit-optimal roll angle over the visible
            time steps, in degrees on [-180, 180).  NaN where the target
            is never visible, and everywhere when no star tracker
            constraint is active — the roll search only runs when one is.
        roll_spread : ndarray, shape (n_dec, n_ra)
            Circular standard deviation of that roll angle, in degrees.
            Small where one roll serves the whole window; large where the
            trackers force a different attitude from orbit to orbit, which
            is where ``mean_roll`` stops meaning much.
        times : the time grid, echoed back.
        step_minutes : float
            Grid spacing in minutes.
        elapsed_seconds : float
            Wall-clock time the sweep took.
    """
    step_minutes = (times[1] - times[0]).to_value(u.min)
    n_ra, n_dec = len(ra_vals), len(dec_vals)
    n_points = n_ra * n_dec

    duty = np.zeros((n_dec, n_ra))
    boresight_duty = np.zeros((n_dec, n_ra))
    longest_window = np.zeros((n_dec, n_ra))
    n_windows = np.zeros((n_dec, n_ra), dtype=int)
    mean_roll = np.full((n_dec, n_ra), np.nan)
    roll_spread = np.full((n_dec, n_ra), np.nan)

    ra_grid, dec_grid = np.meshgrid(ra_vals, dec_vals)
    coords = SkyCoord(ra=ra_grid.ravel() * u.deg, dec=dec_grid.ravel() * u.deg,
                      frame="icrs")

    started = _clock.time()
    report_every = max(n_points // 10, 1)

    for index in range(n_points):
        result = visibility.get_visibility_best_roll(
            coords[index], times,
            roll_step=roll_step, orbit_time_step=orbit_time_step,
        )
        visible = np.asarray(result["visible"], dtype=bool)
        runs = _run_lengths(visible)

        row, column = divmod(index, n_ra)
        duty[row, column] = 100.0 * visible.mean()
        boresight_duty[row, column] = 100.0 * np.asarray(
            result["boresight_visible"], dtype=bool
        ).mean()
        longest_window[row, column] = (
            runs.max() * step_minutes if runs.size else 0.0
        )
        n_windows[row, column] = runs.size

        # The roll angle is circular, so it is averaged as a unit vector:
        # the arithmetic mean of -179 and +179 deg would be 0, which is
        # the opposite attitude.  The resultant's length also gives the
        # spread for free.
        roll_deg = np.asarray(result["roll_deg"], dtype=float)
        chosen = roll_deg[np.isfinite(roll_deg)]
        if chosen.size:
            resultant = np.exp(1j * np.deg2rad(chosen)).mean()
            mean_roll[row, column] = np.rad2deg(np.angle(resultant))
            # Identical angles give a resultant length of 1 give or take
            # rounding, and 1 + 1e-16 would send the log positive and the
            # sqrt to NaN.  Clip both ends before taking either.
            length = float(np.clip(abs(resultant), 1e-12, 1.0))
            roll_spread[row, column] = np.rad2deg(
                np.sqrt(-2.0 * np.log(length))
            )

        # One line per tenth rather than a carriage-returned counter: a
        # saved notebook keeps every flush as its own output cell, and
        # a redrawn progress bar turns into a wall of them.
        if progress and (index == n_points - 1
                         or (index and index % report_every == 0)):
            done = index + 1
            elapsed = _clock.time() - started
            remaining = elapsed * (n_points - done) / done
            print(f"  {done:>6} / {n_points} grid points "
                  f"({100 * done / n_points:5.1f}%) — "
                  f"{_format_duration(elapsed)} elapsed, "
                  f"~{_format_duration(remaining)} left")

    return {
        "ra_vals": ra_vals,
        "dec_vals": dec_vals,
        "duty": duty,
        "boresight_duty": boresight_duty,
        "longest_window": longest_window,
        "n_windows": n_windows,
        "mean_roll": mean_roll,
        "roll_spread": roll_spread,
        "times": times,
        "step_minutes": step_minutes,
        "elapsed_seconds": _clock.time() - started,
    }


def body_radec(name, time, visibility=None):
    """
    ICRS RA/Dec of a solar system body, for marking on the sky maps.

    Parameters
    ----------
    name : str
        Body name understood by ``astropy.coordinates.get_body``.
    time : astropy.time.Time
        Scalar time to evaluate at.
    visibility : pandoravisibility.Visibility, optional
        When given, the body is evaluated from the spacecraft rather than
        the geocentre.  Only matters for the Moon, and only at the ~1°
        level.

    Returns
    -------
    ra_deg, dec_deg : float

    Notes
    -----
    ``get_body`` returns a GCRS coordinate, whose RA/Dec are read off
    directly.  Converting it to ICRS would move the origin to the solar
    system barycentre and give the body's position *there*, which for the
    Sun is meaningless; the GCRS direction differs from ICRS only by
    aberration, well under an arcminute.
    """
    location = None
    if visibility is not None:
        location = visibility._get_observer_location(time)
    coord = get_body(name, time=time, location=location)
    return float(coord.ra.deg % 360.0), float(coord.dec.deg)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _duty_levels(step=10.0, vmax=100.0):
    """Contour levels from 0 to *vmax* per cent, every *step* per cent."""
    return np.arange(0.0, vmax + step / 2, step)


def _label_outline():
    """Dark stroke that keeps white labels legible over any colour map."""
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=2.4, foreground="0.15")]


def _attach_colorbar(ax, mappable, label, ticks=None):
    """
    Colour bar matched to the drawn height of an equal-aspect axes.

    ``fig.colorbar(ax=ax)`` sizes itself from the axes' allotted box,
    which the equal aspect leaves much taller than the axes ends up, so
    the bar would run well past both edges of the map.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.18)
    bar = ax.figure.colorbar(mappable, cax=cax, ticks=ticks)
    bar.set_label(label)
    bar.outline.set_visible(False)
    return bar


def _format_sky_axes(ax, markers=None):
    """
    Equal-aspect RA/Dec axes with optional annotated marker points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    markers : dict, optional
        ``label -> (ra_deg, dec_deg)`` points to annotate.
    """
    outline = _label_outline()
    for label, (ra_deg, dec_deg) in (markers or {}).items():
        ax.plot(ra_deg % 360.0, dec_deg, marker="o", markersize=8,
                markerfacecolor="none", markeredgecolor="white",
                markeredgewidth=1.5, zorder=5, path_effects=outline)
        ax.annotate(label, (ra_deg % 360.0, dec_deg),
                    textcoords="offset points", xytext=(10, -4),
                    color="white", fontsize=9, zorder=5,
                    path_effects=outline)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0.0, 361.0, 30.0))
    ax.set_yticks(np.arange(-90.0, 91.0, 30.0))
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.grid(alpha=0.15, linewidth=0.6)


def _sky_figure(results, n_panels):
    """
    Stack of *n_panels* sky axes, with the grid to draw on them.

    360° of RA against 180° of Dec drawn to equal scale makes each panel
    twice as wide as it is tall, plus room for the title.

    Parameters
    ----------
    results : dict
        Output of :func:`duty_cycle_map`.
    n_panels : int
        Number of stacked panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    ra_grid, dec_grid : ndarray
        Meshgrids carrying a repeated RA=0 column at 360, so a map closes
        on the right edge instead of stopping one grid step short.
    close_seam : callable
        Applies the same repeat to a ``(n_dec, n_ra)`` value array.
    """
    import matplotlib.pyplot as plt

    ra_closed = np.append(results["ra_vals"], 360.0)
    ra_grid, dec_grid = np.meshgrid(ra_closed, results["dec_vals"])

    def close_seam(values):
        return np.column_stack([values, values[:, :1]])

    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 5.6 * n_panels),
                             squeeze=False)
    return fig, axes[:, 0], ra_grid, dec_grid, close_seam


def plot_duty_rectangular(results, level_step=10.0, label_step=20.0,
                          vmax=100.0, markers=None, show_windows=True):
    """
    Duty cycle on a plain RA/Dec grid, with labelled contour lines.

    Plain rectangular axes rather than a sky projection: a coordinate can
    be read straight off both scales, and one degree of RA is the same
    length as one degree of Dec, so a circular keep-out looks circular.
    The cost is the usual one — area is stretched towards the poles, so
    the polar caps look far larger than the sky they cover.  The summary
    statistics weight by solid angle and are unaffected.

    A second panel optionally shows the longest continuous visible
    window, which is what limits a single uninterrupted observation.

    Parameters
    ----------
    results : dict
        Output of :func:`duty_cycle_map`.
    level_step : float
        Spacing of the filled contour levels, in per cent.
    label_step : float
        Spacing of the labelled contour lines, in per cent.
    vmax : float
        Top of the colour scale, in per cent.  The default 100 keeps maps
        of different windows directly comparable; in low Earth orbit the
        Earth occultation caps the duty cycle near 50%, so pass something
        like 60 to spread the ramp over the range the data occupies.
    markers : dict, optional
        ``label -> (ra_deg, dec_deg)`` points to annotate.
    show_windows : bool
        Add the longest-continuous-window panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    outline = _label_outline()
    fig, axes, ra_grid, dec_grid, close_seam = _sky_figure(
        results, 2 if show_windows else 1
    )

    duty = close_seam(results["duty"])
    levels = _duty_levels(level_step, vmax)
    ax = axes[0]
    filled = ax.contourf(ra_grid, dec_grid, duty, levels=levels,
                         cmap=DUTY_CMAP, vmin=0.0, vmax=vmax,
                         extend="max" if vmax < 100.0 else "neither")
    lines = ax.contour(ra_grid, dec_grid, duty,
                       levels=_duty_levels(label_step, vmax)[1:-1],
                       colors="white", linewidths=0.8, alpha=0.7)
    for text in ax.clabel(lines, fmt="%.0f%%", fontsize=8, colors="white"):
        text.set_path_effects(outline)
    _attach_colorbar(ax, filled, "Duty cycle (%)", ticks=levels[::2])
    ax.set_title(f"Duty cycle over {_window_label(results)}")

    if show_windows:
        window = close_seam(results["longest_window"]) / 60.0
        ax2 = axes[1]
        top = max(np.ceil(window.max() * 10) / 10, 0.5)
        window_levels = np.linspace(0.0, top, 11)
        filled2 = ax2.contourf(ra_grid, dec_grid, window, levels=window_levels,
                               cmap=WINDOW_CMAP)
        _attach_colorbar(ax2, filled2, "Longest window (hours)")
        ax2.set_title("Longest continuous visible window")

    for ax in axes:
        _format_sky_axes(ax, markers)

    fig.tight_layout()
    return fig


def plot_roll_map(results, markers=None, show_spread=True, spread_max=90.0):
    """
    Mean orbit-optimal roll angle at each RA/Dec, and how much it varies.

    The roll angle is the one ``get_visibility_best_roll`` settles on for
    each orbit: measured from the projection of celestial north onto the
    plane perpendicular to the boresight, positive right-handed about the
    boresight, reported on [-180, 180).  It is held fixed for a whole
    orbit, so a one-day window contributes about fifteen values per
    coordinate, each weighted here by how long it was actually in use.

    Drawn with ``pcolormesh`` and a cyclic colour map rather than filled
    contours.  -180 and +180 are the same attitude, so contouring would
    interpolate a full sweep of false intermediate values across that
    branch cut, and a normal ramp would print the two ends as opposites.

    Parameters
    ----------
    results : dict
        Output of :func:`duty_cycle_map`.
    markers : dict, optional
        ``label -> (ra_deg, dec_deg)`` points to annotate.
    show_spread : bool
        Add the circular-standard-deviation panel.  A mean roll only says
        something where the spread is small, so this is on by default.
    spread_max : float
        Top of the spread colour scale, in degrees.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Grey marks coordinates with no roll angle at all — either never
    visible in this window, or no star tracker keep-out was configured,
    in which case ``get_visibility_best_roll`` never runs a roll search
    and every entry is NaN.
    """
    from matplotlib import colormaps

    fig, axes, ra_grid, dec_grid, close_seam = _sky_figure(
        results, 2 if show_spread else 1
    )

    mean_roll = close_seam(results["mean_roll"])
    if not np.isfinite(mean_roll).any():
        raise ValueError(
            "no roll angles were selected — get_visibility_best_roll only "
            "searches for one when a star tracker keep-out is active"
        )

    cyclic = colormaps[ROLL_CMAP].with_extremes(bad=NO_DATA_COLOR)
    ax = axes[0]
    mesh = ax.pcolormesh(ra_grid, dec_grid, np.ma.masked_invalid(mean_roll),
                         cmap=cyclic, vmin=-180.0, vmax=180.0,
                         shading="nearest")
    _attach_colorbar(ax, mesh, "Mean roll angle (deg)",
                     ticks=np.arange(-180.0, 181.0, 60.0))
    ax.set_title(f"Mean orbit-optimal roll over {_window_label(results)}")

    if show_spread:
        spread = close_seam(results["roll_spread"])
        sequential = colormaps[SPREAD_CMAP].with_extremes(bad=NO_DATA_COLOR)
        ax2 = axes[1]
        mesh2 = ax2.pcolormesh(ra_grid, dec_grid, np.ma.masked_invalid(spread),
                               cmap=sequential, vmin=0.0, vmax=spread_max,
                               shading="nearest")
        _attach_colorbar(ax2, mesh2, "Roll spread (deg)")
        ax2.set_title("Circular standard deviation of the roll angle")

    for ax in axes:
        _format_sky_axes(ax, markers)

    fig.tight_layout()
    return fig


def roll_summary(results):
    """
    Text summary of the roll angles the sweep selected.

    Parameters
    ----------
    results : dict
        Output of :func:`duty_cycle_map`.

    Returns
    -------
    dict
        n_with_roll : grid points that got a roll angle at all.
        circular_mean : circular mean roll over those points, degrees.
        median_spread : median circular standard deviation, degrees.
        fraction_single_roll : fraction of those points whose roll varied
            by less than 5 deg across the whole window, i.e. where one
            fixed attitude would have served.
    """
    mean_roll = results["mean_roll"]
    spread = results["roll_spread"]
    valid = np.isfinite(mean_roll)
    if not valid.any():
        return {
            "n_with_roll": 0,
            "circular_mean": float("nan"),
            "median_spread": float("nan"),
            "fraction_single_roll": float("nan"),
        }

    resultant = np.exp(1j * np.deg2rad(mean_roll[valid])).mean()
    return {
        "n_with_roll": int(valid.sum()),
        "circular_mean": float(np.rad2deg(np.angle(resultant))),
        "median_spread": float(np.median(spread[valid])),
        "fraction_single_roll": float((spread[valid] < 5.0).mean()),
    }


def _window_label(results):
    """``"1.0 d from 2026-08-03 00:00 UTC"``, for plot titles."""
    times = results["times"]
    span_days = (times[-1] - times[0]).to_value(u.day) + \
        results["step_minutes"] / (24.0 * 60.0)
    return f"{span_days:.2f} d from {times[0].iso[:16]} UTC"


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def sky_area_weights(dec_vals, n_ra):
    """
    Solid angle of each grid cell, normalised to sum to one.

    A plain mean over the grid over-counts the poles, where the RA columns
    crowd together.  Weighting each row by the band of sky it represents,
    ``sin(dec + d/2) - sin(dec - d/2)``, fixes that, so "half the sky" in
    the summary means half the sky.

    Parameters
    ----------
    dec_vals : ndarray, shape (n_dec,)
        Declination axis in degrees.
    n_ra : int
        Number of RA columns.

    Returns
    -------
    ndarray, shape (n_dec, n_ra)
        Weights summing to 1.
    """
    dec = np.deg2rad(dec_vals)
    edges = np.empty(len(dec) + 1)
    edges[1:-1] = 0.5 * (dec[:-1] + dec[1:])
    edges[0] = -np.pi / 2
    edges[-1] = np.pi / 2
    band = np.sin(edges[1:]) - np.sin(edges[:-1])
    weights = np.repeat(band[:, None], n_ra, axis=1) / n_ra
    return weights / weights.sum()


def _weighted_quantile(values, weights, quantiles):
    """Weighted quantiles of *values*, with *weights* summing to one."""
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def duty_summary(results, thresholds=(10.0, 25.0, 40.0, 50.0)):
    """
    Area-weighted summary of the duty-cycle map.

    Parameters
    ----------
    results : dict
        Output of :func:`duty_cycle_map`.
    thresholds : sequence of float
        Duty-cycle levels, in per cent, to report the sky fraction above.

    Returns
    -------
    dict
        mean, median, p10, p90, best, worst : duty cycle in per cent.
        fraction_above : ``threshold -> fraction of the sky`` (0 to 1).
        fraction_never : fraction of the sky never observable.
        best_ra, best_dec : coordinates of the best grid point, degrees.
        mean_boresight : area-weighted mean of the boresight-only duty
            cycle, for comparison with ``mean``.
        median_longest_window : area-weighted median longest window, in
            minutes, over the sky that is observable at all.
    """
    duty = results["duty"]
    weights = sky_area_weights(results["dec_vals"], len(results["ra_vals"]))
    flat_duty = duty.ravel()
    flat_weights = weights.ravel()

    p10, median, p90 = _weighted_quantile(flat_duty, flat_weights,
                                          [0.10, 0.50, 0.90])
    best_index = np.unravel_index(np.argmax(duty), duty.shape)

    observable = flat_duty > 0
    if observable.any():
        median_window = _weighted_quantile(
            results["longest_window"].ravel()[observable],
            flat_weights[observable], [0.50],
        )[0]
    else:
        median_window = 0.0

    return {
        "mean": float((flat_duty * flat_weights).sum()),
        "median": float(median),
        "p10": float(p10),
        "p90": float(p90),
        "best": float(duty[best_index]),
        "worst": float(duty.min()),
        "fraction_above": {
            float(level): float(flat_weights[flat_duty >= level].sum())
            for level in thresholds
        },
        "fraction_never": float(flat_weights[flat_duty <= 0].sum()),
        "best_ra": float(results["ra_vals"][best_index[1]]),
        "best_dec": float(results["dec_vals"][best_index[0]]),
        "mean_boresight": float(
            (results["boresight_duty"].ravel() * flat_weights).sum()
        ),
        "median_longest_window": float(median_window),
    }


def plot_duty_distribution(results, ax=None):
    """
    Fraction of the sky reaching at least a given duty cycle.

    Reads as "x per cent of the sky is observable at least y per cent of
    the window", which is the scheduling question the map answers
    pointwise.

    Parameters
    ----------
    results : dict
        Output of :func:`duty_cycle_map`.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    weights = sky_area_weights(results["dec_vals"], len(results["ra_vals"]))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.2))

    for values, color, label in (
        (results["duty"], "#2b6cb0", "all constraints"),
        (results["boresight_duty"], "#a0aec0", "boresight only"),
    ):
        flat = values.ravel()
        order = np.argsort(flat)[::-1]
        ax.plot(100.0 * np.cumsum(weights.ravel()[order]), flat[order],
                linewidth=2.0, color=color, label=label)

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Fraction of the sky (%)")
    ax.set_ylabel("Duty cycle reached or exceeded (%)")
    ax.set_title(f"Sky coverage over {_window_label(results)}")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax
