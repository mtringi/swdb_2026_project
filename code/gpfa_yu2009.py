"""Gaussian-process factor analysis for NWB-style unit tables.

This implements the latent-state model described by Yu et al. (2009):

    y_t = C x_t + d + eps_t,       x ~ GP(0, K),

with an RBF GP prior and diagonal observation noise. The EM updates are
intended for exploratory neural population analyses, not for reproducing
every optimization detail of the original MATLAB toolbox.

Typical use::

    from gpfa_yu2009 import fit_gpfa_units

    result = fit_gpfa_units(
        area1_units,
        trial_windows=trials[["start_time", "stop_time"]].to_numpy(),
        bin_size_s=0.02,
        n_latent=3,
    )
    latent = result.latent_trajectories  # trial x time x latent dimension

``units`` must contain one row per unit and a ``spike_times`` column. Spike
times and trial windows must use the same absolute time base, in seconds.
Without trial windows, the whole recording is treated as one trial.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


@dataclass
class GPFAResult:
    """Outputs from :func:`fit_gpfa_units`."""

    latent_trajectories: np.ndarray
    time: np.ndarray
    spike_counts: np.ndarray
    unit_ids: np.ndarray
    model: "GPFA"


def _condition_groups(condition_labels):
    labels = np.asarray(condition_labels)
    if labels.ndim != 1:
        raise ValueError("condition_labels must be one-dimensional")
    return [(label, np.flatnonzero(labels == label))
            for label in dict.fromkeys(labels.tolist())]


def _condition_colors(labels, colors=None):
    unique_labels = [label for label, _ in _condition_groups(labels)]
    if colors is None:
        palette = plt.get_cmap("tab10")
        colors = {label: palette(index % 10)
                  for index, label in enumerate(unique_labels)}
    else:
        missing = [label for label in unique_labels if label not in colors]
        if missing:
            raise ValueError(f"colors missing labels: {missing}")
    return colors


def plot_trial_variability(result, condition_labels, ax=None,
                           condition_name="condition", confidence=0.95,
                           colors=None):
    """Plot latent dispersion through time and return summary metrics.

    Dispersion is the mean squared Euclidean distance from each trial to its
    condition mean at each time bin. ``summary`` contains the time-averaged
    dispersion for each condition, useful for statistical comparisons.
    """
    trajectories = np.asarray(result.latent_trajectories)
    labels = np.asarray(condition_labels)
    if trajectories.ndim != 3 or len(labels) != trajectories.shape[0]:
        raise ValueError("condition_labels must match the number of trajectories")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    colors = _condition_colors(labels, colors)

    summary = {}
    for label, indices in _condition_groups(labels):
        if len(indices) < 2:
            continue
        group = trajectories[indices]
        mean_trajectory = group.mean(axis=0)
        dispersion = ((group - mean_trajectory) ** 2).sum(axis=2)
        average = dispersion.mean(axis=0)
        standard_error = dispersion.std(axis=0, ddof=1) / np.sqrt(len(indices))
        z_value = 1.96 if confidence == 0.95 else 1.0
        color = colors[label]
        ax.plot(result.time, average, label=str(label), color=color)
        ax.fill_between(
            result.time, average - z_value * standard_error,
            average + z_value * standard_error, color=color, alpha=0.2,
        )
        summary[label] = {
            "n_trials": len(indices),
            "dispersion_by_time": average,
            "mean_dispersion": float(average.mean()),
        }
    ax.set_xlabel("Time in warped trial (s)")
    ax.set_ylabel("Latent trial-to-trial dispersion")
    ax.set_title(f"Trial-to-trial variability by {condition_name}")
    ax.legend(title=condition_name)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    return ax, summary


def plot_latent_trajectories(result, condition_labels, dimensions=(0, 1),
                             ax=None, condition_name="condition",
                             individual_alpha=0.12, colors=None):
    """Plot individual and mean latent trajectories by condition.

    ``dimensions`` selects two latent dimensions for a 2-D trajectory plot.
    The returned ``summary`` contains each condition's mean trajectory and
    integrated path length.
    """
    trajectories = np.asarray(result.latent_trajectories)
    labels = np.asarray(condition_labels)
    if trajectories.ndim != 3 or len(labels) != trajectories.shape[0]:
        raise ValueError("condition_labels must match the number of trajectories")
    if len(dimensions) != 2 or max(dimensions) >= trajectories.shape[2]:
        raise ValueError("dimensions must contain two fitted latent dimensions")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    colors = _condition_colors(labels, colors)

    summary = {}
    for label, indices in _condition_groups(labels):
        if len(indices) == 0:
            continue
        group = trajectories[indices][:, :, dimensions]
        for trajectory in group:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[label],
                alpha=individual_alpha, lw=0.8)
        mean_trajectory = group.mean(axis=0)
        line = ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1],
                   color=colors[label], lw=2.5, label=str(label))[0]
        ax.scatter(*mean_trajectory[0], color=line.get_color(), marker="o",
                   edgecolor="black", zorder=3)
        ax.scatter(*mean_trajectory[-1], color=line.get_color(), marker="X",
                   edgecolor="black", zorder=3)
        summary[label] = {
            "n_trials": len(indices),
            "mean_trajectory": mean_trajectory,
            "integrated_path_length": float(
                np.linalg.norm(np.diff(mean_trajectory, axis=0), axis=1).sum()
            ),
        }
    ax.set_xlabel(f"Latent dimension {dimensions[0] + 1}")
    ax.set_ylabel(f"Latent dimension {dimensions[1] + 1}")
    ax.set_title(f"Latent trajectories by {condition_name}")
    ax.legend(title=condition_name)
    ax.set_aspect("equal", adjustable="datalim")
    return ax, summary


def compute_trajectory_speed(result, condition_labels=None):
    """Compute latent speed for each trial and optional condition summaries.

    Speed is Euclidean latent distance per second. For warped fits, the time
    unit is seconds in the normalized trial, so speed is normalized-trial
    speed rather than absolute recording-time speed.
    """
    trajectories = np.asarray(result.latent_trajectories)
    time = np.asarray(result.time, dtype=float)
    if trajectories.ndim != 3 or time.ndim != 1:
        raise ValueError("result must contain 3-D trajectories and 1-D time")
    if trajectories.shape[1] != len(time) or len(time) < 2:
        raise ValueError("result.time must match trajectory bins and have length >= 2")
    time_steps = np.diff(time)
    if np.any(time_steps <= 0):
        raise ValueError("result.time must be strictly increasing")

    speed = np.linalg.norm(np.diff(trajectories, axis=1), axis=2) / time_steps[None, :]
    output = {
        "speed_by_trial": speed,
        "speed_time": (time[:-1] + time[1:]) / 2,
        "mean_speed_by_trial": speed.mean(axis=1),
    }
    if condition_labels is not None:
        labels = np.asarray(condition_labels)
        if len(labels) != trajectories.shape[0]:
            raise ValueError("condition_labels must match the number of trials")
        output["by_condition"] = {
            label: {
                "n_trials": int(np.sum(labels == label)),
                "mean_speed": speed[labels == label].mean(axis=0),
                "mean_speed_over_time": float(speed[labels == label].mean()),
            }
            for label in dict.fromkeys(labels.tolist())
        }
    return output


def plot_trajectory_speed(result, condition_labels, ax=None,
                          condition_name="condition", colors=None):
    """Plot mean latent trajectory speed and standard error by condition."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    speed_result = compute_trajectory_speed(result, condition_labels)
    speed_time = speed_result["speed_time"]
    labels = np.asarray(condition_labels)
    colors = _condition_colors(labels, colors)
    for label, values in speed_result["by_condition"].items():
        group_speed = speed_result["speed_by_trial"][labels == label]
        standard_error = group_speed.std(axis=0, ddof=1) / np.sqrt(len(group_speed))
        color = colors[label]
        ax.plot(speed_time, values["mean_speed"], label=str(label), color=color)
        ax.fill_between(
            speed_time,
            values["mean_speed"] - standard_error,
            values["mean_speed"] + standard_error,
            color=color,
            alpha=0.2,
        )
    ax.set_xlabel("Time in warped trial (s)")
    ax.set_ylabel("Latent trajectory speed (units/s)")
    ax.set_title(f"Trajectory speed by {condition_name}")
    ax.legend(title=condition_name)
    return ax, speed_result


def _as_windows(spike_times, trial_windows, bin_size_s):
    if trial_windows is not None:
        windows = np.asarray(trial_windows, dtype=float)
        if windows.ndim != 2 or windows.shape[1] != 2:
            raise ValueError("trial_windows must have shape (n_trials, 2)")
        if np.any(windows[:, 1] <= windows[:, 0]):
            raise ValueError("each trial window must have stop > start")
        return windows

    nonempty = [np.asarray(times, dtype=float) for times in spike_times if len(times)]
    if not nonempty:
        raise ValueError("spike_times contains no spikes and no trial_windows were given")
    start = min(times.min() for times in nonempty)
    stop = max(times.max() for times in nonempty)
    stop = start + max(bin_size_s, np.ceil((stop - start) / bin_size_s) * bin_size_s)
    return np.array([[start, stop]], dtype=float)


def trial_windows_from_dataframe(trials, start_column="start_time",
                                 stop_column="stop_time", onset_column=None,
                                 t_pre_s=None, t_post_s=None):
    """Return absolute windows from an NWB trials table.

    If ``onset_column`` is supplied, windows are defined relative to that
    event, from ``onset + t_pre_s`` through ``onset + t_post_s``.
    """
    if onset_column is not None:
        if t_pre_s is None or t_post_s is None:
            raise ValueError("t_pre_s and t_post_s are required with onset_column")
        if t_post_s <= t_pre_s:
            raise ValueError("t_post_s must be greater than t_pre_s")
        if onset_column not in trials.columns:
            raise KeyError(f"trials is missing required column: {onset_column}")
        onset = trials[onset_column].to_numpy(dtype=float)
        windows = np.column_stack((onset + t_pre_s, onset + t_post_s))
    else:
        missing = [column for column in (start_column, stop_column)
                   if column not in trials.columns]
        if missing:
            raise KeyError(f"trials is missing required column(s): {missing}")
        windows = trials[[start_column, stop_column]].to_numpy(dtype=float)
    if not np.isfinite(windows).all():
        raise ValueError("trial timing columns contain non-finite values")
    if np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("each trial must have stop_time > start_time")
    return windows


def units_to_counts(units, trial_windows=None, bin_size_s=0.02,
                    spike_column="spike_times", trials=None,
                    trial_start_column="start_time",
                    trial_stop_column="stop_time", warp_trials=False,
                    warp_duration_s=None, trial_onset_column=None,
                    t_pre_s=None, t_post_s=None):
    """Convert one-row-per-unit spike times into trial x bin x unit counts."""
    if spike_column not in units.columns:
        raise KeyError(f"units must contain a '{spike_column}' column")
    spike_times = [np.asarray(value, dtype=float) for value in units[spike_column]]
    if any(times.ndim != 1 for times in spike_times):
        raise ValueError(f"each {spike_column} value must be a one-dimensional array")
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")

    if trials is not None:
        if trial_windows is not None:
            raise ValueError("provide either trials or trial_windows, not both")
        trial_windows = trial_windows_from_dataframe(
            trials, trial_start_column, trial_stop_column,
            onset_column=trial_onset_column, t_pre_s=t_pre_s, t_post_s=t_post_s,
        )
    windows = _as_windows(spike_times, trial_windows, bin_size_s)
    n_bins = int(np.ceil((windows[:, 1] - windows[:, 0]).max() / bin_size_s))
    durations = windows[:, 1] - windows[:, 0]
    if warp_trials:
        target_duration = (float(np.median(durations)) if warp_duration_s is None
                           else float(warp_duration_s))
        if target_duration <= 0:
            raise ValueError("warp_duration_s must be positive")
        n_bins = int(np.ceil(target_duration / bin_size_s))
    elif not np.allclose(durations, n_bins * bin_size_s):
        raise ValueError("trial windows must have equal durations")

    counts = np.zeros((len(windows), n_bins, len(spike_times)), dtype=float)
    for trial_index, (start, stop) in enumerate(windows):
        if warp_trials:
            edges = np.arange(n_bins + 1) * bin_size_s
            scale = (edges[-1] / (stop - start))
        else:
            edges = start + np.arange(n_bins + 1) * bin_size_s
            scale = 1.0
        for unit_index, times in enumerate(spike_times):
            trial_spikes = times[(times >= start) & (times <= stop)]
            if warp_trials:
                trial_spikes = (trial_spikes - start) * scale
            counts[trial_index, :, unit_index] = np.histogram(trial_spikes, edges)[0]
    time = (np.arange(n_bins) + 0.5) * bin_size_s
    if trial_onset_column is not None:
        time += t_pre_s
    return counts, time


class GPFA:
    """Expectation-maximization estimator for a compact GPFA model."""

    def __init__(self, n_latent=3, bin_size_s=0.02, tau_s=0.1,
                 max_iter=100, tol=1e-4, sqrt_transform=True):
        if n_latent < 1 or tau_s <= 0 or max_iter < 1:
            raise ValueError("n_latent, tau_s, and max_iter must be positive")
        self.n_latent = int(n_latent)
        self.bin_size_s = float(bin_size_s)
        self.tau_s = float(tau_s)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.sqrt_transform = bool(sqrt_transform)

    def fit(self, spike_counts):
        """Fit to counts with shape ``(trials, time_bins, units)``."""
        counts = np.asarray(spike_counts, dtype=float)
        if counts.ndim != 3 or np.any(counts < 0):
            raise ValueError("spike_counts must be a non-negative 3-D array")
        n_trials, n_time, n_units = counts.shape
        q = min(self.n_latent, n_units, n_trials * n_time)
        observations = np.sqrt(counts) if self.sqrt_transform else counts / self.bin_size_s
        self.mean_ = observations.mean(axis=(0, 1))
        centered = observations - self.mean_
        _, _, components = np.linalg.svd(
            centered.reshape(-1, n_units), full_matrices=False
        )
        self.loading_ = components[:q].T.copy()
        residual = centered - centered @ self.loading_ @ self.loading_.T
        self.noise_variance_ = np.maximum(residual.var(axis=(0, 1)), 1e-5)

        time_delta = np.arange(n_time)[:, None] - np.arange(n_time)[None, :]
        kernel = np.exp(-0.5 * (time_delta * self.bin_size_s / self.tau_s) ** 2)
        kernel += 1e-6 * np.eye(n_time)
        kernel_inverse = np.linalg.inv(kernel)
        latent_observation_precision = self.loading_.T @ (
            (1.0 / self.noise_variance_)[:, None] * self.loading_
        )
        temporal_precision = np.kron(kernel_inverse, np.eye(q))
        observation_precision = np.kron(
            np.eye(n_time), latent_observation_precision
        )
        previous_loss = np.inf

        for _ in range(self.max_iter):
            inv_noise = 1.0 / self.noise_variance_
            precision = temporal_precision + observation_precision
            posterior_cov = np.linalg.inv(precision)
            posterior = np.empty((n_trials, n_time, q))
            for trial_index in range(n_trials):
                residual = observations[trial_index] - self.mean_
                right_hand_side = (residual @ (inv_noise[:, None] * self.loading_)).reshape(-1)
                posterior[trial_index] = solve(precision, right_hand_side,
                                               assume_a="pos").reshape(n_time, q)

            posterior_cov_diag = np.array([
                posterior_cov[
                    time_index * q:(time_index + 1) * q,
                    time_index * q:(time_index + 1) * q,
                ]
                for time_index in range(n_time)
            ])
            expected_xx = posterior_cov_diag[None] + np.einsum(
                "tpi,tpj->tpij", posterior, posterior
            )
            total_xx = expected_xx.sum(axis=(0, 1))
            self.loading_ = np.einsum(
                "tpi,tpj->ij", observations - self.mean_, posterior
            ) @ np.linalg.inv(total_xx + 1e-6 * np.eye(q))
            self.mean_ = (observations - posterior @ self.loading_.T).mean(axis=(0, 1))
            residual = observations - self.mean_ - posterior @ self.loading_.T
            uncertainty = np.einsum(
                "ij,tjk,ik->ti", self.loading_, posterior_cov_diag, self.loading_
            )
            self.noise_variance_ = np.maximum(
                (residual ** 2).mean(axis=(0, 1)) + uncertainty.mean(axis=0), 1e-5
            )
            loss = float(np.mean(residual ** 2))
            if abs(previous_loss - loss) <= self.tol * max(previous_loss, 1.0):
                break
            previous_loss = loss

        self.n_latent_ = q
        self.kernel_ = kernel
        self.bin_count_ = n_time
        self.latent_trajectories_ = self.transform(counts)
        return self

    def transform(self, spike_counts):
        """Infer posterior latent means for new counts with this model."""
        counts = np.asarray(spike_counts, dtype=float)
        if counts.ndim != 3 or counts.shape[1] != self.bin_count_:
            raise ValueError("new counts must have shape (trials, fitted_time_bins, units)")
        observations = np.sqrt(counts) if self.sqrt_transform else counts / self.bin_size_s
        inv_noise = 1.0 / self.noise_variance_
        latent_observation_precision = self.loading_.T @ (
            inv_noise[:, None] * self.loading_
        )
        precision = np.kron(np.linalg.inv(self.kernel_), np.eye(self.n_latent_))
        precision += np.kron(np.eye(self.bin_count_), latent_observation_precision)
        posterior = np.empty((len(observations), self.bin_count_, self.n_latent_))
        for trial_index in range(len(observations)):
            right_hand_side = (
                (observations[trial_index] - self.mean_)
                @ (inv_noise[:, None] * self.loading_)
            ).reshape(-1)
            posterior[trial_index] = solve(
                precision,
                right_hand_side,
                assume_a="pos",
            ).reshape(self.bin_count_, self.n_latent_)
        return posterior


def fit_gpfa_units(units, trial_windows=None, bin_size_s=0.02, n_latent=3,
                   tau_s=0.1, max_iter=100, tol=1e-4, sqrt_transform=True,
                   spike_column="spike_times", trials=None,
                   trial_start_column="start_time",
                   trial_stop_column="stop_time", warp_trials=False,
                   warp_duration_s=None, trial_onset_column=None,
                   t_pre_s=None, t_post_s=None):
    """Bin units and fit GPFA, optionally deriving and warping trial windows."""
    counts, time = units_to_counts(
        units, trial_windows=trial_windows, bin_size_s=bin_size_s,
        spike_column=spike_column, trials=trials,
        trial_start_column=trial_start_column,
        trial_stop_column=trial_stop_column, warp_trials=warp_trials,
        warp_duration_s=warp_duration_s, trial_onset_column=trial_onset_column,
        t_pre_s=t_pre_s, t_post_s=t_post_s,
    )
    model = GPFA(
        n_latent=n_latent, bin_size_s=bin_size_s, tau_s=tau_s,
        max_iter=max_iter, tol=tol, sqrt_transform=sqrt_transform,
    ).fit(counts)
    return GPFAResult(
        latent_trajectories=model.latent_trajectories_,
        time=time,
        spike_counts=counts,
        unit_ids=np.asarray(units.index),
        model=model,
    )