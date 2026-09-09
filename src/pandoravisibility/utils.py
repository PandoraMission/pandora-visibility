"""
Extended utilities for pandoravisibility analysis.

This module provides higher-level analysis functions that build on the core
Visibility class for mission planning and target analysis.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .visibility import Visibility

__all__ = [
    "analyze_yearly_visibility",
    "find_continuous_periods",
    "calculate_visibility_statistics",
    "plot_yearly_visibility",
]


def analyze_yearly_visibility(
    tle_line1,
    tle_line2,
    target_coord,
    start_time=None,
    duration_days=365,
    time_resolution_hours=1,
    visibility_threshold_hours=24,
    verbose=False,
):
    """
    Analyze when a target will be continuously visible over a year.

    Parameters:
    -----------
    tle_line1, tle_line2 : str
        TLE lines for satellite
    target_coord : SkyCoord
        Target coordinates
    start_time : Time, optional
        Start time for analysis (default: Time.now())
    duration_days : int
        Duration to analyze in days (default: 365)
    time_resolution_hours : float
        Time resolution for visibility checks in hours (default: 1)
    visibility_threshold_hours : float
        Minimum continuous visibility duration in hours (default: 24)
    verbose : bool
        Print progress information

    Returns:
    --------
    dict
        Dictionary with visibility analysis results
    """

    if start_time is None:
        start_time = Time.now()
    elif not isinstance(start_time, Time):
        # Allow string input and convert to Time
        start_time = Time(start_time)

    if verbose:
        print("Analyzing yearly visibility for target:")
        print(f"  RA: {target_coord.ra:.4f}")
        print(f"  DEC: {target_coord.dec:.4f}")
        print(f"  Start time: {start_time.iso}")
        print(f"  Duration: {duration_days} days")
        print(f"  Time resolution: {time_resolution_hours} hours")

    # Create time array using astropy
    total_hours = duration_days * 24
    n_points = int(total_hours / time_resolution_hours)

    time_deltas = np.arange(0, total_hours, time_resolution_hours) * u.hour
    times = start_time + time_deltas

    if verbose:
        print(f"  Checking {n_points} time points over {duration_days} days")
        print(f"  End time: {times[-1].iso}")

    # Initialize visibility calculator
    vis = Visibility(tle_line1, tle_line2)

    # Calculate visibility for all times
    if verbose:
        print("  Computing visibility...")

    visibility_results = np.asarray(
        vis.get_visibility(target_coord, times)["visible"]
    )

    # Find continuous visibility periods
    continuous_periods = find_continuous_periods(
        times, visibility_results, visibility_threshold_hours
    )

    # Calculate statistics
    stats = calculate_visibility_statistics(
        times, visibility_results, continuous_periods, visibility_threshold_hours
    )

    if verbose:
        print(f"  Found {len(continuous_periods)} continuous visibility periods")
        print(
            f"  Total visible time: {stats['total_visible_hours']:.1f} hours ({stats['visibility_percentage']:.1f}%)"
        )

    results = {
        "times": times,
        "visibility": visibility_results,
        "continuous_periods": continuous_periods,
        "statistics": stats,
        "target_coord": target_coord,
        "analysis_params": {
            "start_time": start_time,
            "duration_days": duration_days,
            "time_resolution_hours": time_resolution_hours,
            "visibility_threshold_hours": visibility_threshold_hours,
        },
    }

    return results


def find_continuous_periods(times, visibility, min_duration_hours):
    """
    Find periods of continuous visibility.

    Parameters:
    -----------
    times : Time array
        Time points
    visibility : bool array
        Visibility results
    min_duration_hours : float
        Minimum duration for a period to be considered

    Returns:
    --------
    list
        List of continuous visibility periods
    """
    periods = []

    # Find transitions
    visible_diff = np.diff(np.concatenate(([False], visibility, [False])).astype(int))

    # Find start and end indices of visible periods
    start_indices = np.where(visible_diff == 1)[0]
    end_indices = np.where(visible_diff == -1)[0]

    for start_idx, end_idx in zip(start_indices, end_indices):
        if end_idx > start_idx:  # Valid period
            start_time = times[start_idx]
            end_time = times[end_idx - 1]  # Last visible time
            duration = (end_time - start_time).to(u.hour)

            if duration.value >= min_duration_hours:
                periods.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": duration.value,
                        "duration_days": duration.to(u.day).value,
                        "start_index": start_idx,
                        "end_index": end_idx - 1,
                    }
                )

    return periods


def calculate_visibility_statistics(
    times, visibility, continuous_periods, min_duration_hours
):
    """Calculate visibility statistics."""

    total_time = times[-1] - times[0]
    total_time_hours = total_time.to(u.hour).value
    total_time_days = total_time.to(u.day).value

    visible_points = np.sum(visibility)
    total_points = len(visibility)

    time_resolution_hours = total_time_hours / total_points
    total_visible_hours = visible_points * time_resolution_hours
    visibility_percentage = (visible_points / total_points) * 100

    # Statistics for continuous periods
    if continuous_periods:
        period_durations = [period["duration_hours"] for period in continuous_periods]
        longest_period_hours = max(period_durations)
        average_period_hours = np.mean(period_durations)
        total_continuous_hours = sum(period_durations)
    else:
        longest_period_hours = 0
        average_period_hours = 0
        total_continuous_hours = 0

    return {
        "total_time_hours": total_time_hours,
        "total_time_days": total_time_days,
        "total_visible_hours": total_visible_hours,
        "visibility_percentage": visibility_percentage,
        "continuous_periods_count": len(continuous_periods),
        "longest_continuous_period_hours": longest_period_hours,
        "longest_continuous_period_days": longest_period_hours / 24,
        "average_continuous_period_hours": average_period_hours,
        "total_continuous_hours": total_continuous_hours,
        "continuous_visibility_percentage": (total_continuous_hours / total_time_hours)
        * 100,
        "time_resolution_hours": time_resolution_hours,
    }


def plot_yearly_visibility(results, figsize=(15, 10)):
    """
    Plot yearly visibility analysis results.

    Parameters:
    -----------
    results : dict
        Results from analyze_yearly_visibility
    figsize : tuple
        Figure size (default: (15, 10))

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    times = results["times"]
    visibility = results["visibility"]
    continuous_periods = results["continuous_periods"]
    stats = results["statistics"]
    params = results["analysis_params"]

    # Convert astropy Time to matplotlib-compatible datetime
    times_datetime = [t.datetime for t in times]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Plot 1: Raw visibility
    axes[0].plot(times_datetime, visibility.astype(int), "b-", linewidth=0.5, alpha=0.7)
    axes[0].fill_between(
        times_datetime, 0, visibility.astype(int), alpha=0.3, color="blue"
    )
    axes[0].set_ylabel("Visible\n(1=Yes, 0=No)")
    axes[0].set_title(
        f"Target Visibility Over Time\n"
        f'RA: {results["target_coord"].ra:.3f}, '
        f'DEC: {results["target_coord"].dec:.3f}'
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # Plot 2: Continuous periods
    y_continuous = np.zeros_like(visibility, dtype=float)
    for period in continuous_periods:
        start_idx = period["start_index"]
        end_idx = period["end_index"]
        y_continuous[start_idx : end_idx + 1] = 1

    axes[1].plot(times_datetime, y_continuous, "r-", linewidth=1)
    axes[1].fill_between(times_datetime, 0, y_continuous, alpha=0.5, color="red")
    axes[1].set_ylabel(
        f'Continuous Periods\n(≥{params["visibility_threshold_hours"]}h)'
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)

    # Add period labels
    for i, period in enumerate(continuous_periods[:10]):  # Label first 10 periods
        mid_time = (
            period["start_time"] + (period["end_time"] - period["start_time"]) / 2
        )
        axes[1].annotate(
            f'{period["duration_days"]:.1f}d',
            xy=(mid_time.datetime, 0.5),
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    # Plot 3: Rolling visibility percentage (3-day window)
    window_days = 3
    window_points = int(window_days * 24 / params["time_resolution_hours"])
    if window_points > 0 and window_points < len(visibility):
        rolling_visibility = (
            np.convolve(
                visibility.astype(float),
                np.ones(window_points) / window_points,
                mode="same",
            )
            * 100
        )
        axes[2].plot(times_datetime, rolling_visibility, "g-", linewidth=1)
        axes[2].fill_between(
            times_datetime, 0, rolling_visibility, alpha=0.3, color="green"
        )
    else:
        # Fallback for short time series
        axes[2].plot(times_datetime, visibility.astype(float) * 100, "g-", linewidth=1)
        axes[2].fill_between(
            times_datetime, 0, visibility.astype(float) * 100, alpha=0.3, color="green"
        )

    axes[2].set_ylabel(f"{window_days}-Day Rolling\nVisibility %")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 105)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    # Add statistics text box
    stats_text = (
        f"Overall Visibility: {stats['visibility_percentage']:.1f}%\n"
        f"Continuous Periods: {stats['continuous_periods_count']}\n"
        f"Longest Period: {stats['longest_continuous_period_days']:.1f} days\n"
        f"Avg Period: {stats['average_continuous_period_hours']:.1f}h"
    )

    axes[0].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[0].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    return fig
