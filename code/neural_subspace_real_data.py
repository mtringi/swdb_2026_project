"""
neural_subspace_real_data.py

Same PCA state-space / single-trial-projection analysis originally developed
in `neural_subspace_projection.py`, adapted to run on REAL recorded data
instead of the synthetic simulator. All subspace-fitting and plotting code
is self-contained in this file (no dependency on the synthetic-data script);
only the data-generation step is replaced with a loader for two tables:

  1. A SPIKE-TIMES table: one row per (unit, trial), with a column holding
     an array of spike times (in seconds, ALIGNED to trial onset -- i.e.
     t=0 is stimulus/trial onset for every trial, negative values allowed
     for a pre-stimulus baseline window).

  2. A TRIAL table: one row per trial, with (at minimum) a stimulus
     identity and stimulus modality column.

============================================================================
WHAT YOU NEED TO PROVIDE, AND WHY
============================================================================
You mentioned you have spike times + stimulus id/modality. That's enough
for the most basic version of this analysis (a subspace built from
stimulus-conditioned PSTHs), but to reproduce the full set of comparisons
from the synthetic version -- hit vs. miss vs. correct-reject vs. false
alarm, and "correct vs. incorrect response to the same physical stimulus"
-- the trial table needs a few more columns. Concretely:

  REQUIRED (you said you have these):
    - trial_id        unique trial identifier, shared with the spike table
    - stim_id         stimulus identity (e.g. 'vis_target', 'aud_target', ...)
    - stim_modality    'visual' or 'auditory' (or your own two labels)

  NEEDED FOR THE FULL OUTCOME / CORRECT-VS-INCORRECT COMPARISONS:
    - context          the task's rewarded/relevant modality for that
                        trial (i.e. which block the animal was in --
                        'visual' or 'auditory'). This is NOT the same as
                        stim_modality: on a false-alarm trial, stim_modality
                        can differ from context. If your task has no block
                        structure, set this equal to stim_modality and the
                        "context-gated" comparisons will just be inert.
    - licked           bool: whether the animal responded on that trial
    - is_go            bool: whether that trial was a go trial (i.e. the
                        stimulus was the rewarded target given the current
                        context). If you don't have this directly, you can
                        usually derive it from stim_id/context if your task
                        follows a "lick to target-in-matching-context" rule
                        (see `infer_is_go` below for an example / template
                        you'll need to adapt to your own task logic).

  If `context` / `licked` / `is_go` are missing, the script still runs --
  it just skips the outcome-based and correct-vs-incorrect plots and only
  produces the stimulus-conditioned subspace + trajectories.

  ALSO NEEDED:
    - bin_size_s, and the peri-trial window (t_pre, t_post) your spike
      times were aligned/cut to -- these must match what you used when you
      extracted the spike_times arrays, so the code bins them correctly.

Only numpy / scipy / pandas / matplotlib are used.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d


# ==========================================================================
# 0. Spikes -> smoothed single-trial firing rates
# ==========================================================================

def spikes_to_rates(spike_counts, bin_size_s, smooth_sigma_bins=2.0, sqrt_transform=False):
    """(n_trials, n_time, n_neurons) spike counts -> smoothed rate (Hz).

    sqrt_transform=True applies the variance-stabilizing sqrt transform of
    raw counts (Yu et al. 2009; Cunningham & Yu, 2014) before smoothing,
    as in Perich et al. 2018 -- in which case the output is smoothed
    sqrt(counts), not Hz."""
    if sqrt_transform:
        rates = np.sqrt(spike_counts.astype(float))
    else:
        rates = spike_counts.astype(float) / bin_size_s
    if smooth_sigma_bins > 0:
        rates = gaussian_filter1d(rates, sigma=smooth_sigma_bins, axis=1)
    return rates


# ==========================================================================
# 1. Neural subspace: PCA on condition-averaged (denoised) PSTHs
# ==========================================================================

class NeuralSubspace:
    """PCA subspace fit on z-scored, condition-averaged population rates.
    Single trials are later projected into this fixed subspace."""

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.components_ = None                # (n_neurons, n_components)
        self.explained_variance_ratio_ = None

    def fit(self, condition_avg_rates):
        """condition_avg_rates: (n_conditions, n_time, n_neurons)."""
        n_cond, n_time, n_neurons = condition_avg_rates.shape
        flat = condition_avg_rates.reshape(-1, n_neurons)

        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        z = (flat - self.mean_) / self.std_

        # PCA via SVD (no sklearn dependency)
        _, S, Vt = np.linalg.svd(z, full_matrices=False)
        var = (S ** 2) / max(z.shape[0] - 1, 1)
        self.explained_variance_ratio_ = var / var.sum()
        self.components_ = Vt[: self.n_components].T
        return self

    def transform(self, rates):
        """rates: (..., n_neurons) -> (..., n_components)."""
        z = (rates - self.mean_) / self.std_
        return z @ self.components_


def build_condition_averages(rates, keys, min_trials=5):
    """rates: (n_trials, n_time, n_neurons); keys: list of hashable trial
    labels (e.g. (stim, context, outcome) tuples). Returns the stacked
    condition-average rate trajectories and the corresponding keys, dropping
    any condition with fewer than `min_trials` trials (too noisy to average)."""
    keys = [tuple(k) if isinstance(k, (tuple, list)) else k for k in keys]
    uniq = sorted(set(keys))
    avgs, kept_keys, counts = [], [], []
    for k in uniq:
        mask = np.array([row == k for row in keys])
        n = mask.sum()
        if n >= min_trials:
            avgs.append(rates[mask].mean(axis=0))
            kept_keys.append(k)
            counts.append(int(n))
    return np.stack(avgs, axis=0), kept_keys, counts


# ==========================================================================
# 2. Simple task-axis (choice axis): mean-difference direction, a
#    lightweight analogue of Mante's regression-based targeted axes.
# ==========================================================================

def mean_difference_axis(rates, group_a_mask, group_b_mask, mean_, std_):
    """Direction (unit vector, n_neurons,) separating the time-and-trial
    averaged activity of two trial groups, in z-scored coordinates."""
    za = (rates[group_a_mask] - mean_) / std_
    zb = (rates[group_b_mask] - mean_) / std_
    diff = za.mean(axis=(0, 1)) - zb.mean(axis=(0, 1))
    norm = np.linalg.norm(diff)
    return diff / norm if norm > 0 else diff


def label_outcomes(data):
    """hit / miss / correct_reject / false_alarm, from is_go & licked."""
    is_go, licked = data["is_go"], data["licked"]
    outcome = np.full(len(is_go), "", dtype=object)
    outcome[is_go & licked] = "hit"
    outcome[is_go & ~licked] = "miss"
    outcome[~is_go & ~licked] = "correct_reject"
    outcome[~is_go & licked] = "false_alarm"
    return outcome


# ==========================================================================
# 3. Plotting helpers
# ==========================================================================

OUTCOME_COLORS = {
    "hit": "#1b9e77",
    "miss": "#7570b3",
    "correct_reject": "#377eb8",
    "false_alarm": "#d95f02",
}
OUTCOME_LABELS = {
    "hit": "Hit (correct lick)",
    "miss": "Miss",
    "correct_reject": "Correct reject",
    "false_alarm": "False alarm",
}


def _plot_trajectory(ax, traj, color, lw=2.0, alpha=1.0, label=None, marker_end=True):
    """traj: (n_time, 2) trajectory in PC1-PC2 (or any 2D) space."""
    ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, alpha=alpha, label=label)
    ax.scatter(*traj[0], color=color, marker="o", s=45, zorder=5,
                edgecolor="k", linewidth=0.5)
    if marker_end:
        ax.scatter(*traj[-1], color=color, marker="X", s=60, zorder=5,
                    edgecolor="k", linewidth=0.5)


def plot_outcome_trajectories(pc_trials, outcome, pcx=0, pcy=1,
                                title="", save_path=None):
    """Average + all single-trial trajectories per outcome category."""
    fig, ax = plt.subplots(figsize=(6.5, 6))

    for oc in ["correct_reject", "hit", "false_alarm", "miss"]:
        mask = np.where(outcome == oc)[0]
        if len(mask) == 0:
            continue
        color = OUTCOME_COLORS[oc]

        # light single-trial traces (all trials in this outcome category)
        for idx in mask:
            ax.plot(pc_trials[idx, :, pcx], pc_trials[idx, :, pcy],
                     color=color, lw=0.5, alpha=0.15)

        # bold condition-average trajectory
        avg = pc_trials[mask].mean(axis=0)
        _plot_trajectory(ax, avg[:, [pcx, pcy]], color=color, lw=3.0,
                          label=f"{OUTCOME_LABELS[oc]}  (n={len(mask)})")

    ax.set_xlabel(f"PC{pcx + 1}")
    ax.set_ylabel(f"PC{pcy + 1}")
    ax.set_title(title)
    legend_elems = [Line2D([0], [0], marker="o", color="k", label="trial start",
                            markerfacecolor="w", markersize=7, linestyle="None"),
                    Line2D([0], [0], marker="X", color="k", label="trial end",
                            markerfacecolor="w", markersize=8, linestyle="None")]
    leg1 = ax.legend(loc="upper left", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    ax.axhline(0, color="grey", lw=0.5, zorder=0)
    ax.axvline(0, color="grey", lw=0.5, zorder=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_stim_correct_vs_incorrect(pc_trials, correct_mask, incorrect_mask,
                                     time_bins, stim_label, correct_label,
                                     incorrect_label, pcx=0, pcy=1,
                                     title="", save_path=None):
    """Generic version: compare trajectories for one physical stimulus type
    presented correctly (matched context, go response = hit) vs. incorrectly
    (mismatched context, go response = false alarm)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    ax = axes[0]
    groups = [(correct_label, correct_mask, "#1b9e77"),
              (incorrect_label, incorrect_mask, "#d95f02")]
    for label, mask_idx, color in groups:
        if len(mask_idx) == 0:
            continue
        for idx in mask_idx:
            ax.plot(pc_trials[idx, :, pcx], pc_trials[idx, :, pcy],
                     color=color, lw=0.5, alpha=0.15)
        avg = pc_trials[mask_idx].mean(axis=0)
        _plot_trajectory(ax, avg[:, [pcx, pcy]], color=color, lw=3.0,
                          label=f"{label}  (n={len(mask_idx)})")
    ax.set_xlabel(f"PC{pcx + 1}")
    ax.set_ylabel(f"PC{pcy + 1}")
    ax.set_title(f"Same physical stimulus ({stim_label}),\ndifferent context/outcome")
    ax.legend(loc="best", fontsize=8)
    ax.axhline(0, color="grey", lw=0.5, zorder=0)
    ax.axvline(0, color="grey", lw=0.5, zorder=0)

    # right panel: trajectory separation (Euclidean distance) over time,
    # with a trial-label permutation null distribution
    ax2 = axes[1]
    avg_c = pc_trials[correct_mask].mean(axis=0)
    avg_i = pc_trials[incorrect_mask].mean(axis=0)
    n_comp = pc_trials.shape[-1]
    dist = np.linalg.norm(avg_c[:, :n_comp] - avg_i[:, :n_comp], axis=1)

    all_idx = np.concatenate([correct_mask, incorrect_mask])
    n_c = len(correct_mask)
    null_dists = []
    rng2 = np.random.default_rng(2)
    for _ in range(200):
        perm = rng2.permutation(all_idx)
        a = pc_trials[perm[:n_c]].mean(axis=0)
        b = pc_trials[perm[n_c:]].mean(axis=0)
        null_dists.append(np.linalg.norm(a[:, :n_comp] - b[:, :n_comp], axis=1))
    null_dists = np.array(null_dists)
    null_hi = np.percentile(null_dists, 97.5, axis=0)
    null_lo = np.percentile(null_dists, 2.5, axis=0)

    ax2.plot(time_bins, dist, color="k", lw=2.0, label="observed distance")
    ax2.fill_between(time_bins, null_lo, null_hi, color="grey", alpha=0.3,
                       label="null 95% CI (shuffled labels)")
    ax2.set_xlabel("Time in trial (s)")
    ax2.set_ylabel(f"Euclidean distance in PC1-{n_comp} space")
    ax2.set_title(f"Correct vs. incorrect {stim_label} trajectory separation")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_trajectory_distance(pc_trials, correct_mask, incorrect_mask, time_bins,
                               stim_label, correct_label, incorrect_label,
                               n_pcs=None, title="", save_path=None):
    """Standalone version of the right-hand panel of
    `plot_stim_correct_vs_incorrect`: Euclidean distance between the
    correct/incorrect condition-average trajectories over time, with a
    trial-label permutation null. `n_pcs` controls how many of the fitted
    PCs (columns of `pc_trials`) the distance is computed over -- pass e.g.
    10 to compare across the top 10 PCs instead of whatever was used for
    the 2D trajectory plots (`pc_trials` must have >= n_pcs columns)."""
    n_comp = pc_trials.shape[-1] if n_pcs is None else n_pcs
    avg_c = pc_trials[correct_mask].mean(axis=0)
    avg_i = pc_trials[incorrect_mask].mean(axis=0)
    dist = np.linalg.norm(avg_c[:, :n_comp] - avg_i[:, :n_comp], axis=1)

    all_idx = np.concatenate([correct_mask, incorrect_mask])
    n_c = len(correct_mask)
    null_dists = []
    rng2 = np.random.default_rng(2)
    for _ in range(200):
        perm = rng2.permutation(all_idx)
        a = pc_trials[perm[:n_c]].mean(axis=0)
        b = pc_trials[perm[n_c:]].mean(axis=0)
        null_dists.append(np.linalg.norm(a[:, :n_comp] - b[:, :n_comp], axis=1))
    null_dists = np.array(null_dists)
    null_hi = np.percentile(null_dists, 97.5, axis=0)
    null_lo = np.percentile(null_dists, 2.5, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(time_bins, dist, color="k", lw=2.0, label="observed distance")
    ax.fill_between(time_bins, null_lo, null_hi, color="grey", alpha=0.3,
                      label="null 95% CI (shuffled labels)")
    ax.set_xlabel("Time in trial (s)")
    ax.set_ylabel(f"Euclidean distance in PC1-{n_comp} space")
    ax.set_title(title or
                 f"Correct vs. incorrect {stim_label} trajectory separation "
                 f"(top {n_comp} PCs)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def find_elbow(explained_variance_ratio):
    """Locate the 'elbow'/plateau of a scree curve using the standard
    distance-to-line heuristic: draw a straight line from the first point
    to the last point of the curve, and pick the component with the
    largest perpendicular distance from that line. That's the point where
    the curve bends over from steeply dropping to flat."""
    y = np.asarray(explained_variance_ratio, dtype=float)
    n = len(y)
    if n <= 2:
        return n
    x = np.arange(1, n + 1, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p1, p2 = np.array([x_n[0], y_n[0]]), np.array([x_n[-1], y_n[-1]])
    line_dir = (p2 - p1) / np.linalg.norm(p2 - p1)
    vecs = np.stack([x_n, y_n], axis=1) - p1
    proj = np.outer(vecs @ line_dir, line_dir)
    dist = np.linalg.norm(vecs - proj, axis=1)
    return int(np.argmax(dist)) + 1  # 1-indexed PC number


def plot_scree(subspace, save_path=None, pad_components=4, min_shown=6):
    """Scree plot, cropped to just past where the curve plateaus (rather
    than showing every PC, which is unreadable with many neurons/units).
    The full spectrum is still summarized in the title."""
    evr = subspace.explained_variance_ratio_
    n_total = len(evr)
    elbow = find_elbow(evr)
    n_show = min(n_total, max(elbow + pad_components, min_shown))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(1, n_show + 1)
    ax.bar(x, evr[:n_show], color="#4c72b0")
    ax.plot(x, np.cumsum(evr)[:n_show], "o-", color="k", label="cumulative")
    ax.axvline(elbow, color="crimson", linestyle="--", lw=1.3,
                label=f"plateau ~ PC{elbow}")
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance explained")
    subtitle = (f"showing PC1-{n_show} of {n_total} total"
                if n_show < n_total else f"all {n_total} PCs shown")
    ax.set_title(f"Subspace variance explained (condition-averaged PSTHs)\n{subtitle}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_choice_axis(pc_trials, choice_proj, outcome, time_bins, title="", save_path=None):
    """1-D projection onto the mean-difference 'choice axis', averaged per
    outcome, +/- SEM across trials."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for oc in ["correct_reject", "hit", "false_alarm", "miss"]:
        mask = np.where(outcome == oc)[0]
        if len(mask) == 0:
            continue
        vals = choice_proj[mask]
        m = vals.mean(axis=0)
        sem = vals.std(axis=0, ddof=1) / np.sqrt(len(mask)) if len(mask) > 1 else np.zeros_like(m)
        ax.plot(time_bins, m, color=OUTCOME_COLORS[oc], lw=2.2,
                 label=f"{OUTCOME_LABELS[oc]} (n={len(mask)})")
        ax.fill_between(time_bins, m - sem, m + sem, color=OUTCOME_COLORS[oc], alpha=0.2)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Time in trial (s)")
    ax.set_ylabel("Projection onto choice axis (a.u.)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


# ==========================================================================
# 4. USER CONFIGURATION -- edit this section for your data
# ==========================================================================

CONFIG = dict(
    # --- file paths (or set to None and pass DataFrames directly instead) ---
    spike_table_path="path/to/your_spike_times_table.csv",   # or .parquet/.pkl
    trial_table_path="path/to/your_trial_table.csv",

    # --- spike time format ---
    # "session": one row per UNIT, spike_times = that unit's full-session
    #     spike times in absolute seconds (e.g. the standard Allen
    #     Institute Neuropixels 'units' table format).
    # "trial_aligned": one row per (unit, trial), spike_times already cut
    #     and aligned so t=0 = that trial's onset.
    spike_time_format="session",

    # --- column name mapping: your column name -> internal name ---
    # Edit the VALUES (your actual column names) on the right; keep the KEYS.
    # Defaults below are pre-filled for the Allen Institute Dynamic Routing
    # units + behavior tables; adjust for your own schema as needed.
    spike_table_columns=dict(
        unit_id="unit_id",
        trial_id="trial_id",        # only used when spike_time_format == "trial_aligned"
        spike_times="spike_times",
    ),
    trial_table_columns=dict(
        trial_id="trial_index",
        stim_id="stim_name",
        # Set stim_modality to a column name if you have one directly, or
        # to None to derive it from is_aud_stim/is_vis_stim flag columns
        # instead (as below) -- only one of the two paths is used.
        stim_modality=None,
        is_aud_stim="is_aud_stim",       # only used when stim_modality is None
        is_vis_stim="is_vis_stim",       # only used when stim_modality is None
        context="rewarded_modality",     # the block's rewarded/relevant modality
        licked="is_response",            # only used if outcome_columns is None
        is_go="is_go",                   # only used if outcome_columns is None
        # only used when spike_time_format == "session": the trial's
        # alignment/onset time, in the SAME absolute time base as the
        # units table's spike_times.
        trial_onset_time="stim_start_time",
    ),
    # The derived-modality labels MUST exactly match the string values used
    # in your context column (e.g. rewarded_modality). Check with
    # `trial_df['rewarded_modality'].unique()` and edit if these don't match
    # -- the script will also print a warning at runtime if they disagree.
    modality_labels=("aud", "vis"),   # (aud_label, vis_label)

    # If your trial table already has precomputed per-trial outcome
    # columns (as the Dynamic Routing behavior table does), point to them
    # here -- they'll be used directly instead of being derived from
    # is_go/licked, which is preferred since it matches the task's actual
    # scoring exactly. Set to None to derive from is_go/licked instead.
    outcome_columns=dict(
        hit="is_hit",
        miss="is_miss",
        correct_reject="is_correct_reject",
        false_alarm="is_false_alarm",
    ),

    # --- binning: MUST match how your spike_times arrays were aligned/cut ---
    t_pre_s=-0.5,          # start of window relative to trial onset (s); negative = before onset
    t_post_s=1.5,         # end of window relative to trial onset (s)
    bin_size_s=0.1,
    smooth_sigma_bins=0.02,   # gaussian smoothing sigma, in BINS (0 = no smoothing)
    sqrt_transform=False,     # sqrt-transform raw counts before smoothing (Cunningham & Yu, 2014)

    # --- analysis parameters ---
    n_components=3,
    n_components_compare=10,  # extra PCs kept around for the top-N distance comparison plot
    min_trials_per_condition=5,
    out_dir="/tmp",
)


# ==========================================================================
# 5. Load & validate the two tables
# ==========================================================================

def _read_table(path_or_df):
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
    path = str(path_or_df)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unrecognized file extension for {path!r}; "
                      f"pass a DataFrame directly instead, or add a loader.")


def load_tables(spike_table, trial_table):
    """spike_table / trial_table: file paths OR already-loaded DataFrames."""
    return _read_table(spike_table), _read_table(trial_table)


def validate_columns(df, colmap, table_name):
    missing = [internal for internal, actual in colmap.items()
               if actual is not None and actual not in df.columns]
    if missing:
        raise KeyError(
            f"{table_name} is missing column(s) {missing} "
            f"(as mapped in CONFIG). Available columns: {list(df.columns)}"
        )


# ==========================================================================
# 6. Bin trial-aligned spike times into a (n_trials, n_time, n_units) array
# ==========================================================================

def bin_spike_times_to_counts(spike_df, trial_ids, unit_ids, bin_edges,
                                unit_col, trial_col, spike_col):
    """spike_df: one row per (unit, trial), with `spike_col` holding an
    array/list of spike times (s), ALREADY aligned so t=0 = trial onset.

    trial_ids, unit_ids: the exact ordered lists defining axis 0 / axis 2
    of the returned array (so it lines up with your trial table's row
    order and a chosen, consistent unit ordering).

    Returns spike_counts: (n_trials, n_bins, n_units) integer array.
    """
    n_trials, n_units, n_bins = len(trial_ids), len(unit_ids), len(bin_edges) - 1
    counts = np.zeros((n_trials, n_bins, n_units), dtype=int)

    trial_index = {tid: i for i, tid in enumerate(trial_ids)}
    unit_index = {uid: i for i, uid in enumerate(unit_ids)}

    n_skipped = 0
    for row in spike_df.itertuples(index=False):
        tid = getattr(row, trial_col)
        uid = getattr(row, unit_col)
        if tid not in trial_index or uid not in unit_index:
            n_skipped += 1
            continue
        spikes = np.asarray(getattr(row, spike_col), dtype=float)
        spikes = spikes[np.isfinite(spikes)]
        c, _ = np.histogram(spikes, bins=bin_edges)
        counts[trial_index[tid], :, unit_index[uid]] = c

    if n_skipped:
        print(f"  (skipped {n_skipped} spike-table rows with unit/trial ids "
              f"not found in the trial table / unit list)")
    return counts


def bin_session_spike_times_to_counts(units_df, trial_onsets, unit_ids, bin_edges,
                                        unit_col="unit_id", spike_col="spike_times"):
    """For the common Allen-Institute-style 'units' table format: ONE ROW
    PER UNIT, with `spike_col` holding that unit's FULL-SESSION spike times
    (absolute time, seconds) -- NOT trial-aligned, and with no trial_id
    column at all (since the table isn't organized by trial).

    trial_onsets: array-like of per-trial alignment times (s), in the SAME
    absolute time base as spike_col, one per trial, in the same order as
    your trial table's rows (e.g. each trial's stimulus/go-cue onset time).

    Returns spike_counts: (n_trials, n_bins, n_units) integer array.

    Note: this loops over (unit x trial), using np.searchsorted to quickly
    narrow down each unit's spike train to the relevant window before
    histogramming, which is fast enough for typical session sizes (10s-100s
    of units x 100s-1000s of trials).
    """
    n_trials, n_units, n_bins = len(trial_onsets), len(unit_ids), len(bin_edges) - 1
    counts = np.zeros((n_trials, n_bins, n_units), dtype=int)
    trial_onsets = np.asarray(trial_onsets, dtype=float)

    spikes_by_unit = {}
    for row in units_df.itertuples(index=False):
        uid = getattr(row, unit_col)
        spikes_by_unit[uid] = np.sort(np.asarray(getattr(row, spike_col), dtype=float))

    n_missing_units = 0
    for ui, uid in enumerate(unit_ids):
        spikes = spikes_by_unit.get(uid)
        if spikes is None or len(spikes) == 0:
            n_missing_units += 1
            continue
        for ti, onset in enumerate(trial_onsets):
            local_edges = bin_edges + onset
            i0, i1 = np.searchsorted(spikes, [local_edges[0], local_edges[-1]])
            c, _ = np.histogram(spikes[i0:i1], bins=local_edges)
            counts[ti, :, ui] = c

    if n_missing_units:
        print(f"  (found no spikes for {n_missing_units} of {n_units} unit_ids "
              f"-- check unit_id values match between tables)")
    return counts


def align_spike_times_to_trials(raw_spike_df, trial_onsets, unit_col, trial_col,
                                  spike_col, onset_time_col="trial_onset_s"):
    """OPTIONAL helper: if your spike times are in ABSOLUTE session time
    (not already trial-aligned), use this first to subtract each trial's
    onset time, producing the trial-aligned long format that
    `bin_spike_times_to_counts` expects.

    raw_spike_df: one row per (unit, trial) with spike_col = absolute-time
    spike arrays that fall within that trial's window.
    trial_onsets: dict/Series mapping trial_id -> onset time (s).
    """
    df = raw_spike_df.copy()
    onset = df[trial_col].map(trial_onsets)
    df[spike_col] = [np.asarray(s, dtype=float) - t0
                       for s, t0 in zip(df[spike_col], onset)]
    return df


# ==========================================================================
# 7. Build the trial-metadata arrays the subspace/plotting code expects
# ==========================================================================

def infer_is_go(stim_id, stim_modality, context):
    """TEMPLATE ONLY -- adapt to your own task's go/no-go rule.
    As written, this assumes the Dynamic-Routing convention: a trial is a
    go trial if the stimulus is the *target* stimulus for the modality
    that matches the current context. This will not be correct for other
    task designs; if `is_go` is available directly in your trial table, use
    that instead and ignore this function entirely."""
    is_target = np.array([str(s).endswith("target") and "non" not in str(s)
                           for s in stim_id])
    modality_matches = np.asarray(stim_modality) == np.asarray(context)
    return is_target & modality_matches


def derive_stim_modality(trial_df, is_aud_col, is_vis_col, aud_label, vis_label):
    """Build a stim_modality array from two boolean flag columns (e.g. the
    Allen Dynamic Routing behavior table's is_aud_stim / is_vis_stim),
    rather than requiring a single modality-string column. `aud_label` /
    `vis_label` should match whatever string values your context column
    (e.g. rewarded_modality) actually uses, so that stim_modality == context
    comparisons work correctly -- check with e.g. trial_df['rewarded_
    modality'].unique() and adjust CONFIG['modality_labels'] if needed."""
    is_aud = trial_df[is_aud_col].to_numpy().astype(bool)
    is_vis = trial_df[is_vis_col].to_numpy().astype(bool)
    modality = np.full(len(trial_df), "", dtype=object)
    modality[is_aud] = aud_label
    modality[is_vis] = vis_label
    return modality


def build_outcome_from_columns(trial_df, outcome_columns):
    """Use PRECOMPUTED outcome boolean columns directly -- e.g. the Allen
    Dynamic Routing behavior table already provides is_hit / is_false_alarm
    / is_correct_reject / is_miss -- rather than re-deriving outcome from
    is_go/licked. Preferred whenever your trial table already has these,
    since it reflects the task's actual scoring logic exactly rather than
    the generic go/lick assumption in `infer_is_go`."""
    n = len(trial_df)
    outcome = np.full(n, "", dtype=object)
    for label, col in outcome_columns.items():
        outcome[trial_df[col].to_numpy().astype(bool)] = label
    return outcome


def build_trial_arrays(trial_df, colmap, bin_edges, outcome_columns=None,
                         modality_labels=("aud", "vis")):
    stim_id = trial_df[colmap["stim_id"]].to_numpy()

    if colmap.get("stim_modality"):
        stim_modality = trial_df[colmap["stim_modality"]].to_numpy()
    else:
        aud_label, vis_label = modality_labels
        stim_modality = derive_stim_modality(
            trial_df, colmap["is_aud_stim"], colmap["is_vis_stim"],
            aud_label=aud_label, vis_label=vis_label)

    context = (trial_df[colmap["context"]].to_numpy()
               if colmap.get("context") else None)
    licked = (trial_df[colmap["licked"]].to_numpy().astype(bool)
              if colmap.get("licked") else None)
    is_go = (trial_df[colmap["is_go"]].to_numpy().astype(bool)
             if colmap.get("is_go") else None)

    outcome = (build_outcome_from_columns(trial_df, outcome_columns)
               if outcome_columns else None)

    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return dict(stim_id=stim_id, stim_modality=stim_modality, context=context,
                licked=licked, is_go=is_go, outcome=outcome, time_bins=time_bins)


# ==========================================================================
# 8. Main pipeline
# ==========================================================================

def run_real_data_analysis(config=CONFIG, spike_table=None, trial_table=None):
    cfg = config
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    spike_df, trial_df = load_tables(
        spike_table if spike_table is not None else cfg["spike_table_path"],
        trial_table if trial_table is not None else cfg["trial_table_path"],
    )

    sc = cfg["spike_table_columns"]
    tc = cfg["trial_table_columns"]
    spike_format = cfg.get("spike_time_format", "trial_aligned")

    spike_required = ("unit_id", "spike_times") if spike_format == "session" else tuple(sc.keys())
    validate_columns(spike_df, {k: v for k, v in sc.items() if k in spike_required},
                       "spike table")
    validate_columns(trial_df, tc, "trial table")
    if cfg.get("outcome_columns"):
        validate_columns(trial_df, cfg["outcome_columns"], "trial table (outcome_columns)")

    trial_ids = trial_df[tc["trial_id"]].tolist()
    unit_ids = sorted(spike_df[sc["unit_id"]].unique().tolist())
    n_neurons = len(unit_ids)
    print(f"Loaded {len(trial_ids)} trials x {n_neurons} units.")

    bin_edges = np.arange(cfg["t_pre_s"], cfg["t_post_s"] + cfg["bin_size_s"],
                            cfg["bin_size_s"])

    print("Binning spike times into per-trial, per-unit spike counts...")
    if spike_format == "session":
        trial_onsets = trial_df[tc["trial_onset_time"]].to_numpy()
        spike_counts = bin_session_spike_times_to_counts(
            spike_df, trial_onsets, unit_ids, bin_edges,
            unit_col=sc["unit_id"], spike_col=sc["spike_times"],
        )
    else:
        spike_counts = bin_spike_times_to_counts(
            spike_df, trial_ids, unit_ids, bin_edges,
            unit_col=sc["unit_id"], trial_col=sc["trial_id"], spike_col=sc["spike_times"],
        )

    meta = build_trial_arrays(trial_df, tc, bin_edges,
                                outcome_columns=cfg.get("outcome_columns"),
                                modality_labels=cfg.get("modality_labels", ("aud", "vis")))
    time_bins = meta["time_bins"]
    stim_id = meta["stim_id"]
    stim_modality = meta["stim_modality"]
    context = meta["context"]
    licked = meta["licked"]
    is_go = meta["is_go"]
    outcome = meta["outcome"]

    if context is not None and outcome is not None:
        # ignore '' -- that's the expected label for trials that are
        # neither auditory nor visual (e.g. catch/no-stim trials), not a
        # real labeling mismatch.
        nonempty_modalities = set(np.unique(stim_modality)) - {""}
        mismatch = nonempty_modalities - set(np.unique(context))
        if mismatch:
            print(f"WARNING: stim_modality values {mismatch} never appear in your "
                  f"context column's values {set(np.unique(context))}. The "
                  f"correct-vs-incorrect comparison compares these with '==', so "
                  f"if they use different label conventions (e.g. 'vis' vs "
                  f"'visual') that comparison will silently find nothing. Check "
                  f"CONFIG['modality_labels'] against your context column's "
                  f"actual values.")

    have_outcomes = (outcome is not None) or (
        (context is not None) and (licked is not None) and (is_go is not None))
    if not have_outcomes:
        print("NOTE: no outcome_columns provided, and context / licked / is_go "
              "not all provided -- skipping outcome-based and "
              "correct-vs-incorrect comparisons. Only the stimulus-conditioned "
              "subspace will be built.")

    print("Converting spike counts to smoothed firing rates...")
    rates = spikes_to_rates(spike_counts, bin_size_s=cfg["bin_size_s"],
                              smooth_sigma_bins=cfg["smooth_sigma_bins"],
                              sqrt_transform=cfg.get("sqrt_transform", False))

    # ---- 8a. Build the subspace from condition-averaged PSTHs.
    # Use (stim_id, context, outcome) as conditions if outcome info is
    # available; otherwise fall back to stim_id alone.
    if have_outcomes:
        if outcome is None:  # derive from is_go/licked since no outcome_columns given
            outcome = label_outcomes(dict(is_go=is_go, licked=licked))
        condition_keys = list(zip(stim_id.tolist(), context.tolist(), outcome.tolist()))
    else:
        condition_keys = list(stim_id.tolist())

    cond_avg_rates, cond_keys, cond_counts = build_condition_averages(
        rates, condition_keys, min_trials=cfg["min_trials_per_condition"])
    if len(cond_keys) < 2:
        raise RuntimeError(
            "Fewer than 2 conditions have >= min_trials_per_condition trials; "
            "lower min_trials_per_condition or check your trial labels."
        )
    print(f"Fitting PCA subspace on {len(cond_keys)} condition-averaged PSTHs "
          f"(n_components={cfg['n_components']})...")
    # Fit with enough components to cover both the primary (n_components)
    # analyses and the top-N PC distance-comparison plot below; PCA component
    # order/values don't change when more components are requested, so this
    # doesn't alter any of the original n_components-based plots.
    n_fit = max(cfg["n_components"], cfg.get("n_components_compare", 0))
    subspace = NeuralSubspace(n_components=n_fit).fit(cond_avg_rates)
    evr = subspace.explained_variance_ratio_
    print("  variance explained by top PCs:",
          np.round(evr[:cfg["n_components"]], 3),
          " cumulative:", round(evr[:cfg["n_components"]].sum(), 3))

    print("Projecting individual single trials into the subspace...")
    pc_trials_full = subspace.transform(rates)
    pc_trials = pc_trials_full[..., :cfg["n_components"]]

    plot_scree(subspace, save_path=os.path.join(out_dir, "real_subspace_scree.png"))

    if not have_outcomes:
        print(f"Done (stimulus-only analysis). Figures written to {out_dir}/real_subspace_*.png")
        return dict(rates=rates, subspace=subspace, pc_trials=pc_trials,
                     stim_id=stim_id, time_bins=time_bins)

    # ---- 8b. Outcome comparison: hit / miss / correct_reject / false_alarm
    plot_outcome_trajectories(
        pc_trials, outcome, pcx=0, pcy=1,
        title="Single-trial trajectories by outcome (real data)",
        save_path=os.path.join(out_dir, "real_subspace_outcomes.png"),
    )

    # ---- 8c. Correct vs. incorrect response to a fixed physical stimulus,
    # generalized across whatever distinct stim_modality/stim_id values
    # exist in your data (not hard-coded to 'aud_target'/'vis_target').
    unique_stims = sorted(set(zip(stim_id.tolist(), meta["stim_modality"].tolist())))
    for stim, modality in unique_stims:
        is_this_stim = (stim_id == stim)
        matched_ctx = is_this_stim & (context == modality) & (outcome == "hit")
        mismatched_ctx = is_this_stim & (context != modality) & (outcome == "false_alarm")
        if len(matched_ctx.nonzero()[0]) >= 2 and len(mismatched_ctx.nonzero()[0]) >= 2:
            correct_idx = np.where(matched_ctx)[0]
            incorrect_idx = np.where(mismatched_ctx)[0]
            print(f"stim={stim!r}: correct (context-matched hit) n={len(correct_idx)}, "
                  f"incorrect (context-mismatched false alarm) n={len(incorrect_idx)}")
            plot_stim_correct_vs_incorrect(
                pc_trials, correct_idx, incorrect_idx, time_bins,
                stim_label=str(stim),
                correct_label=f"Correct {stim} (matched context, hit)",
                incorrect_label=f"Incorrect {stim} (mismatched context, false alarm)",
                pcx=0, pcy=1,
                title=f"Correct vs. incorrect {stim} response (real data)",
                save_path=os.path.join(out_dir, f"real_subspace_{stim}_correct_vs_incorrect.png"),
            )

            # Same distance measurement, but computed across the top
            # n_components_compare PCs instead of just n_components, as a
            # separate plot (checks whether separation is confined to the
            # first few PCs or extends into weaker, higher-variance-rank ones).
            n_cmp = min(cfg.get("n_components_compare", 0), pc_trials_full.shape[-1])
            if n_cmp > cfg["n_components"]:
                plot_trajectory_distance(
                    pc_trials_full, correct_idx, incorrect_idx, time_bins,
                    stim_label=str(stim),
                    correct_label=f"Correct {stim} (matched context, hit)",
                    incorrect_label=f"Incorrect {stim} (mismatched context, false alarm)",
                    n_pcs=n_cmp,
                    title=f"Correct vs. incorrect {stim} response, top {n_cmp} PCs (real data)",
                    save_path=os.path.join(
                        out_dir, f"real_subspace_{stim}_correct_vs_incorrect_top{n_cmp}pcs.png"),
                )

    # ---- 8c-2. Hit vs. correct-reject to the same physical stimulus: unlike
    # 8c (which holds the motor output/lick fixed and varies context), this
    # holds context-correctness of the *stimulus* the same conceptual
    # "irrelevant" side but instead contrasts the matched-context go response
    # against the mismatched-context correctly-withheld response -- i.e. it
    # includes the correct-reject trials that 8c intentionally excludes, at
    # the cost of also varying licked/not-licked between the two groups.
    for stim, modality in unique_stims:
        is_this_stim = (stim_id == stim)
        matched_ctx = is_this_stim & (context == modality) & (outcome == "hit")
        mismatched_ctx_cr = is_this_stim & (context != modality) & (outcome == "correct_reject")
        if len(matched_ctx.nonzero()[0]) >= 2 and len(mismatched_ctx_cr.nonzero()[0]) >= 2:
            hit_idx = np.where(matched_ctx)[0]
            cr_idx = np.where(mismatched_ctx_cr)[0]
            print(f"stim={stim!r}: hit (context-matched) n={len(hit_idx)}, "
                  f"correct-reject (context-mismatched) n={len(cr_idx)}")
            plot_stim_correct_vs_incorrect(
                pc_trials, hit_idx, cr_idx, time_bins,
                stim_label=str(stim),
                correct_label=f"Hit {stim} (matched context)",
                incorrect_label=f"Correct reject {stim} (mismatched context)",
                pcx=0, pcy=1,
                title=f"Hit vs. correct-reject {stim} response (real data)",
                save_path=os.path.join(
                    out_dir, f"real_subspace_{stim}_hit_vs_correct_reject.png"),
            )

    # ---- 8c-3. Context modulation of a non-target stimulus's representation,
    # holding *both* physical stimulus and behavioral output fixed: for a
    # stimulus that is correctly rejected regardless of which block it falls
    # in (e.g. aud2/vis2 -- distractors that are never the current block's
    # target), compare correct-reject trials when its own modality matches
    # the block context vs. when it doesn't. Both groups are correct-reject
    # (no lick either way), so any separation here can't be attributed to a
    # difference in overt motor output or in correctness -- it isolates a
    # pure context effect on the same stimulus/response pair. This loop
    # naturally produces plots only for stimuli that have enough
    # correct-reject trials in *both* contexts (i.e. non-target stimuli);
    # target stimuli are essentially never correctly rejected when their
    # modality matches the block, so they drop out on the trial-count check.
    for stim, modality in unique_stims:
        is_this_stim = (stim_id == stim)
        matched_cr = is_this_stim & (context == modality) & (outcome == "correct_reject")
        mismatched_cr = is_this_stim & (context != modality) & (outcome == "correct_reject")
        if len(matched_cr.nonzero()[0]) >= 2 and len(mismatched_cr.nonzero()[0]) >= 2:
            matched_idx = np.where(matched_cr)[0]
            mismatched_idx = np.where(mismatched_cr)[0]
            print(f"stim={stim!r}: correct-reject, matched context n={len(matched_idx)}, "
                  f"correct-reject, mismatched context n={len(mismatched_idx)}")
            plot_stim_correct_vs_incorrect(
                pc_trials, matched_idx, mismatched_idx, time_bins,
                stim_label=str(stim),
                correct_label=f"Correct reject {stim} (matched context)",
                incorrect_label=f"Correct reject {stim} (mismatched context)",
                pcx=0, pcy=1,
                title=f"Context effect on correctly-rejected {stim} response (real data)",
                save_path=os.path.join(
                    out_dir, f"real_subspace_{stim}_correct_reject_context_effect.png"),
            )

    # ---- 8d. Choice axis (lick vs. no-lick mean-difference direction)
    choice_axis = mean_difference_axis(rates, licked, ~licked,
                                         subspace.mean_, subspace.std_)
    z_rates = (rates - subspace.mean_) / subspace.std_
    choice_proj = z_rates @ choice_axis
    plot_choice_axis(
        pc_trials, choice_proj, outcome, time_bins,
        title="Projection onto lick/no-lick 'choice axis' (real data)",
        save_path=os.path.join(out_dir, "real_subspace_choice_axis.png"),
    )

    print(f"Done. Figures written to {out_dir}/real_subspace_*.png")
    return dict(rates=rates, subspace=subspace, pc_trials=pc_trials,
                 stim_id=stim_id, context=context, outcome=outcome,
                 choice_proj=choice_proj, time_bins=time_bins)


if __name__ == "__main__":
    # Edit CONFIG above with your real file paths / column names, then:
    results = run_real_data_analysis(CONFIG)
