# %%
from pathlib import Path

import pandas as pd

# Path to import data from
DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
UNITS_FILE = DATA_DIR / "good_units.csv"
# UNITS_FILE = DATA_DIR / "aud_units_session2.csv"
# UNITS_FILE = DATA_DIR / "cp_units_session2.csv"
good_units = pd.read_csv(UNITS_FILE)

# %%
good_units.head(3)
# %%
# Check the columns in the good_units DataFrame
good_units.columns
# %%
# Check the data types in the good_units DataFrame
good_units.dtypes
# %%
# Check the shape of the good_units DataFrame
good_units.shape
# %%
# Check the number of unique values in each column of the good_units DataFrame
for col in good_units.columns:
    print(f"{col}: {good_units[col].nunique()}")
# %%
# Check the distribution of values in each column of the good_units DataFrame
for col in good_units.columns:
    print(f"{col}:")
    print(good_units[col].value_counts())
    print()
# %%
# Check Firing rates for a couple of units, -0.5 to 1.5 s around stimulus onset.
# good_units.csv's own `spike_times` column is truncated, so spikes come from
# the per-stimulus export instead, where `spike_time` is already relative to
# stimulus onset.
import numpy as np
import matplotlib.pyplot as plt

spikes = pd.read_csv(DATA_DIR / "sound1_spike_times.csv")
trials = pd.read_csv(DATA_DIR / "sound1_trials.csv")

TIME_WINDOW = (-0.5, 1.5)
BIN_WIDTH = 0.05
bin_edges = np.arange(TIME_WINDOW[0], TIME_WINDOW[1] + BIN_WIDTH, BIN_WIDTH)
bin_centers = bin_edges[:-1] + BIN_WIDTH / 2
n_trials = trials["id"].nunique()

# %%
# Population-average firing rate for auditory cortex, pooling every region
# whose name contains "AUD" (AUDp, AUDv, AUDpo), one curve per input class
# (sound1 = target, sound2 = non-target).
aud_ids = set(good_units.loc[good_units["structure"].str.contains("AUD", na=False), "id"]) # change this to the title of the file

fig, ax = plt.subplots(figsize=(8, 4))
for stim_name, color in (("sound1", "#2a78d6"), ("sound2", "#eb6834")):
    stim_spikes = pd.read_csv(DATA_DIR / f"{stim_name}_spike_times.csv")
    stim_trials = pd.read_csv(DATA_DIR / f"{stim_name}_trials.csv")
    stim_spikes = stim_spikes[stim_spikes["unit_id"].isin(aud_ids)]

    n_units = stim_spikes["unit_id"].nunique()
    trial_ids = stim_trials["id"].to_numpy()

    # one row per trial: population-average (across AUD units) firing rate
    # over time, so every trial can be drawn as its own faint line
    trial_rates = np.empty((len(trial_ids), len(bin_centers)))
    for i, trial_id in enumerate(trial_ids):
        trial_spike_times = stim_spikes.loc[stim_spikes["trial_id"] == trial_id, "spike_time"]
        counts, _ = np.histogram(trial_spike_times, bins=bin_edges)
        trial_rates[i] = counts / BIN_WIDTH / n_units

    for rate in trial_rates:
        ax.plot(bin_centers, rate, color=color, linewidth=0.6, alpha=0.08, zorder=1)
    ax.plot(bin_centers, trial_rates.mean(axis=0), color=color, linewidth=2, zorder=3,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8,
            label=f"{stim_name} (n={n_units} units, {len(trial_ids)} trials)")

ax.axvline(0, color="grey", linestyle=":", linewidth=1, label="stimulus onset")
ax.set_xlabel("time from stimulus onset (s)")
ax.set_ylabel("firing rate (spikes/s per unit)")
ax.set_title("Auditory cortex (AUDp/AUDv/AUDpo) population average\n"
             "(thin lines: individual trials)")
ax.legend()
plt.show()
# %%
# Now reduce the AUD population activity itself with PCA. Pool sound1 and
# sound2 trials (trial "id" is unique across the whole session - see
# DATA/*_trials.csv - so concatenating them is safe) and bin every AUD unit's
# spikes into the same (trial, time bin) grid used above.
aud_spikes_parts, aud_trials_parts = [], []
for stim_name in ("sound1", "sound2"):
    stim_spikes = pd.read_csv(DATA_DIR / f"{stim_name}_spike_times.csv")
    stim_trials = pd.read_csv(DATA_DIR / f"{stim_name}_trials.csv")
    aud_spikes_parts.append(stim_spikes[stim_spikes["unit_id"].isin(aud_ids)])
    aud_trials_parts.append(stim_trials)

aud_spikes = pd.concat(aud_spikes_parts, ignore_index=True)
aud_trials = (pd.concat(aud_trials_parts, ignore_index=True)
                .sort_values("id").reset_index(drop=True))

trial_id_list = aud_trials["id"].to_numpy()
unit_id_list = np.sort(aud_spikes["unit_id"].unique())
trial_pos = {trial_id: i for i, trial_id in enumerate(trial_id_list)}
unit_pos = {unit_id: i for i, unit_id in enumerate(unit_id_list)}

counts = np.zeros((len(trial_id_list), len(bin_centers), len(unit_id_list)), dtype=np.float32)
trial_idx = aud_spikes["trial_id"].map(trial_pos).to_numpy()
unit_idx = aud_spikes["unit_id"].map(unit_pos).to_numpy()
bin_idx = np.floor((aud_spikes["spike_time"].to_numpy() - bin_edges[0]) / BIN_WIDTH).astype(int)
np.add.at(counts, (trial_idx, bin_idx, unit_idx), 1)
tensor = counts / BIN_WIDTH   # spikes/s

print(f"tensor shape (trials, time bins, units): {tensor.shape}")

# %%
# Processing step: split the trial-level dataset into FIT trials and the
# reward-scheduled ("task-switch") trials that every fit below holds out.
# is_reward_scheduled marks the 5 instruction trials that open each
# aud-rewarded block (3 blocks -> 15 trials, all sound1): the task rule just
# switched under the animal, so their population activity is atypical and they
# should not shape the fitted subspace - they are only ever projected into it
# afterwards, never used to define it.
sched_per_trial = aud_trials["is_reward_scheduled"].to_numpy(dtype=bool)
fit_trial_mask = ~sched_per_trial
print(f"{int(sched_per_trial.sum())} reward-scheduled trials excluded from every fit below; "
      f"{int(fit_trial_mask.sum())} of {len(fit_trial_mask)} trials used to fit")

# %%
# Smooth in time, z-score each unit (so no single high-firing unit dominates),
# then run PCA - the z-scoring statistics AND the PCA subspace are both fit on
# FIT trials only (fit_trial_mask), then applied to EVERY trial, so the
# reward-scheduled trials are purely projected, never fitted.
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

SMOOTHING_SIGMA_BINS = 1.5
N_PCS = 3

smoothed = gaussian_filter1d(tensor, sigma=SMOOTHING_SIGMA_BINS, axis=1, mode="nearest")
unit_means = smoothed[fit_trial_mask].mean(axis=(0, 1))
unit_stds = smoothed[fit_trial_mask].std(axis=(0, 1))
active_units = unit_stds > 0   # drop units that never fire among FIT trials
standardised = (smoothed[:, :, active_units] - unit_means[active_units]) / unit_stds[active_units]

n_trials_pca, n_bins_pca, n_units_pca = standardised.shape
flat_activity = standardised.reshape(n_trials_pca * n_bins_pca, n_units_pca)
flat_fit_mask = np.repeat(fit_trial_mask, n_bins_pca)   # pooled rows from FIT trials only

pca = PCA(n_components=N_PCS)
pca.fit(flat_activity[flat_fit_mask])
scores = pca.transform(flat_activity).reshape(n_trials_pca, n_bins_pca, N_PCS)

print(f"PC1-PC{N_PCS} explain {100 * pca.explained_variance_ratio_.sum():.0f}% of variance "
      f"({', '.join(f'{v:.0%}' for v in pca.explained_variance_ratio_)}) "
      f"[fit on {int(fit_trial_mask.sum())} non-scheduled trials]")

# %%
# PC1 vs PC2 trajectory, all 4 stimulus x context combinations overlaid on one
# axes: colour = stimulus (sound1 / sound2), line style = context (solid =
# auditory-rewarded block, dashed = visual-rewarded block). Sound1 is the lick
# target only in auditory-rewarded blocks; in visual-rewarded blocks both
# sounds are task-irrelevant.
# thin line = one trial, thick line = average, square = trial start,
# diamond = stimulus onset, arrows = direction of time.
stim_name_per_trial = aud_trials["stim_name"].to_numpy()
context_per_trial = aud_trials["rewarded_modality"].to_numpy()
onset_bin = int(np.argmin(np.abs(bin_centers)))
ARROW_EVERY = 4   # bins between arrowheads on the mean trajectory

def add_direction_arrows(ax, path, color, every=ARROW_EVERY):
    """Small arrowheads from path[i-1] to path[i], every `every` bins."""
    for i in range(every, len(path), every):
        ax.annotate("", xy=path[i], xytext=path[i - 1],
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=0, shrinkA=0, shrinkB=0,
                                    mutation_scale=16),
                    zorder=5)

fig, ax = plt.subplots(figsize=(7, 7))
for stim_name, color in (("sound1", "#2a78d6"), ("sound2", "#eb6834")):
    for context, linestyle, context_label in (("aud", "-", "aud-rewarded"),
                                               ("vis", "--", "vis-rewarded")):
        mask = (stim_name_per_trial == stim_name) & (context_per_trial == context)
        if mask.sum() < 2:
            continue
        for trial_scores in scores[mask]:
            ax.plot(trial_scores[:, 0], trial_scores[:, 1], color=color,
                    linewidth=0.6, alpha=0.06, zorder=1)

        mean_trajectory = scores[mask].mean(axis=0)
        ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], color=color, linewidth=2,
                linestyle=linestyle, zorder=3, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"{stim_name}, {context_label} (n={int(mask.sum())})")
        add_direction_arrows(ax, mean_trajectory[:, :2], color)
        ax.plot(mean_trajectory[0, 0], mean_trajectory[0, 1], marker="s", markersize=8,
                color=color, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
        ax.plot(mean_trajectory[onset_bin, 0], mean_trajectory[onset_bin, 1], marker="D",
                markersize=9, color=color, markeredgecolor="white", markeredgewidth=1.5,
                zorder=4)

ax.set_aspect("equal")
ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% var)")
ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}% var)")
ax.set_title("PCA trajectory, auditory cortex (AUDp/AUDv/AUDpo)\n"
             "colour = stimulus, solid/dashed = context, square = start, "
             "diamond = onset, arrows = time")
ax.legend(fontsize=8.5)
fig.tight_layout()
plt.show()
# %%
# PC1 versus each of the first 6 PCs, one 2D panel per pair (PC1 on x, PCk on
# y, k = 2..6). Same conventions as the PC1-vs-PC2 plot above: colour =
# stimulus, solid/dashed = context, thin line = one trial, thick line = the
# mean trajectory, square = trial start, diamond = stimulus onset, arrows =
# time. Aspect is left free here (unlike the main plot) so structure in the
# lower-variance PCs stays visible despite their smaller scale.
#
# Recompute PCA with 6 components here so this cell stands on its own even if
# the PCA cell above was last run with a different N_PCS.
N_PCS_GRID = 6
pca_grid = PCA(n_components=N_PCS_GRID)
pca_grid.fit(flat_activity[flat_fit_mask])   # fit on non-scheduled trials, project everyone
scores_grid = pca_grid.transform(flat_activity).reshape(n_trials_pca, n_bins_pca, N_PCS_GRID)

pc_pairs = [(0, k) for k in range(1, 6)]   # (PC1, PC2) ... (PC1, PC6)
ncols = 3
nrows = int(np.ceil(len(pc_pairs) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
axes_flat = axes.ravel()

for ax, (xpc, ypc) in zip(axes_flat, pc_pairs):
    for stim_name, color in (("sound1", "#2a78d6"), ("sound2", "#eb6834")):
        for context, linestyle, context_label in (("aud", "-", "aud-rewarded"),
                                                   ("vis", "--", "vis-rewarded")):
            mask = (stim_name_per_trial == stim_name) & (context_per_trial == context)
            if mask.sum() < 2:
                continue
            for trial_scores in scores_grid[mask]:
                ax.plot(trial_scores[:, xpc], trial_scores[:, ypc], color=color,
                        linewidth=0.6, alpha=0.06, zorder=1)

            mean_trajectory = scores_grid[mask].mean(axis=0)
            ax.plot(mean_trajectory[:, xpc], mean_trajectory[:, ypc], color=color,
                    linewidth=2, linestyle=linestyle, zorder=3, marker="o",
                    markersize=4, markeredgecolor="white", markeredgewidth=0.8,
                    label=f"{stim_name}, {context_label} (n={int(mask.sum())})")
            add_direction_arrows(ax, mean_trajectory[:, [xpc, ypc]], color)
            ax.plot(mean_trajectory[0, xpc], mean_trajectory[0, ypc], marker="s",
                    markersize=8, color=color, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=4)
            ax.plot(mean_trajectory[onset_bin, xpc], mean_trajectory[onset_bin, ypc],
                    marker="D", markersize=9, color=color, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=4)

    ax.set_xlabel(f"PC{xpc + 1} ({100 * pca_grid.explained_variance_ratio_[xpc]:.0f}% var)")
    ax.set_ylabel(f"PC{ypc + 1} ({100 * pca_grid.explained_variance_ratio_[ypc]:.0f}% var)")

for ax in axes_flat[len(pc_pairs):]:
    ax.set_visible(False)

axes_flat[0].legend(fontsize=8.5)
fig.suptitle("PC1 vs PC2-PC6, auditory cortex (AUDp/AUDv/AUDpo)\n"
             "colour = stimulus, solid/dashed = context, square = start, "
             "diamond = onset, arrows = time")
fig.tight_layout()
plt.show()
# %%
# Every 2D combination of the first 6 PCs, laid out as a lower-triangular pairs
# matrix: column = PC on the x axis (PC1..PC5), row = PC on the y axis
# (PC2..PC6), so panel (row, col) is PC(col+1) vs PC(row+2) and only pairs with
# x-PC < y-PC are drawn (15 panels in total). Every single-trial trajectory is
# drawn faint underneath (SHOW_TRIALS = False for means only), but each panel's
# axis limits are autoscaled to the mean trajectories so the trial cloud does
# not dominate the view. Markers/arrows as before:
# square = trial start, diamond = stimulus onset, arrows = direction of time.
from matplotlib.lines import Line2D

SHOW_TRIALS = True   # overlay every single-trial trajectory; set False for means only
N_PC_MATRIX = 6

pca_mat = PCA(n_components=N_PC_MATRIX)
pca_mat.fit(flat_activity[flat_fit_mask])    # fit on non-scheduled trials, project everyone
scores_mat = pca_mat.transform(flat_activity).reshape(n_trials_pca, n_bins_pca, N_PC_MATRIX)
var_pct = 100 * pca_mat.explained_variance_ratio_

def plot_pc_pair(ax, xpc, ypc):
    """Draw the stimulus x context trajectory set for one (xpc, ypc) PC pair.

    Axis limits are set from the mean trajectories only (with an 8% margin), so
    the faint single-trial cloud is still drawn but does not blow out the view.
    """
    mean_pts = []
    for stim_name, color in (("sound1", "#2a78d6"), ("sound2", "#eb6834")):
        for context, linestyle in (("aud", "-"), ("vis", "--")):
            mask = (stim_name_per_trial == stim_name) & (context_per_trial == context)
            if mask.sum() < 2:
                continue
            if SHOW_TRIALS:
                for trial_scores in scores_mat[mask]:
                    ax.plot(trial_scores[:, xpc], trial_scores[:, ypc], color=color,
                            linewidth=0.5, alpha=0.05, zorder=1)
            m = scores_mat[mask].mean(axis=0)
            mean_pts.append(m[:, [xpc, ypc]])
            ax.plot(m[:, xpc], m[:, ypc], color=color, linewidth=1.8,
                    linestyle=linestyle, zorder=3, marker="o", markersize=3,
                    markeredgecolor="white", markeredgewidth=0.6)
            add_direction_arrows(ax, m[:, [xpc, ypc]], color)
            ax.plot(m[0, xpc], m[0, ypc], marker="s", markersize=7, color=color,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4)
            ax.plot(m[onset_bin, xpc], m[onset_bin, ypc], marker="D", markersize=8,
                    color=color, markeredgecolor="white", markeredgewidth=1.2, zorder=4)

    if mean_pts:
        pts = np.concatenate(mean_pts, axis=0)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        px = 0.08 * ((x1 - x0) or 1.0)
        py = 0.08 * ((y1 - y0) or 1.0)
        ax.set_xlim(x0 - px, x1 + px)
        ax.set_ylim(y0 - py, y1 + py)

side = N_PC_MATRIX - 1
fig, axes = plt.subplots(side, side, figsize=(3.1 * side, 3.1 * side), squeeze=False)

for row in range(side):        # y axis = PC(row + 2)
    for col in range(side):    # x axis = PC(col + 1)
        ax = axes[row, col]
        xpc, ypc = col, row + 1
        if col > row:          # upper triangle: x-PC index >= y-PC index, skip
            ax.set_visible(False)
            continue
        plot_pc_pair(ax, xpc, ypc)
        if col == 0:
            ax.set_ylabel(f"PC{ypc + 1} ({var_pct[ypc]:.0f}%)")
        if row == side - 1:
            ax.set_xlabel(f"PC{xpc + 1} ({var_pct[xpc]:.0f}%)")

legend_handles = [
    Line2D([0], [0], color="#2a78d6", lw=2, label="sound1"),
    Line2D([0], [0], color="#eb6834", lw=2, label="sound2"),
    Line2D([0], [0], color="grey", lw=2, linestyle="-", label="aud-rewarded"),
    Line2D([0], [0], color="grey", lw=2, linestyle="--", label="vis-rewarded"),
    Line2D([0], [0], color="grey", lw=0, marker="s", label="trial start"),
    Line2D([0], [0], color="grey", lw=0, marker="D", label="stimulus onset"),
]
fig.legend(handles=legend_handles, loc="upper right", fontsize=11, frameon=False,
           bbox_to_anchor=(0.98, 0.98))
fig.suptitle("All 2D PC combinations (PC1-PC6), auditory cortex (AUDp/AUDv/AUDpo)\n"
             "colour = stimulus, solid/dashed = context, arrows = time", fontsize=13)
fig.tight_layout()
plt.show()
# %%
# Interactive 3D view of the first 3 PCs, one line per stimulus x context
# combination. Built with Plotly so it can be rotated / zoomed / panned and
# traces toggled from the legend. Running this cell opens the figure and also
# writes a standalone `pca_3d_pc123.html` next to this script that keeps the
# same interactivity (git-ignored).
#
# TRIALS_3D picks the single-trial overlay drawn underneath the means:
#   "none"    - means only
#   "combo"   - every trial, faint, coloured by stimulus x context
#   "outcome" - every trial, green = correct / red = incorrect
#               (is_correct / is_incorrect from the trial table), kept
#               separate for each of the 4 stimulus x context classes -> up to
#               8 legend entries, each toggling on its own. TRIALS_3D_COMBOS
#               optionally limits which classes contribute, e.g.
#               {("sound2", "aud")}.
# Regardless of TRIALS_3D, every single-trial overlay trace starts deselected
# (visible="legendonly") - only the 4 MEAN trajectories are selected when the
# figure first loads; click a legend entry to bring its trials back.
#
# With MARK_REWARD_SCHEDULED, the task-switch trials (is_reward_scheduled: the
# 5 instruction trials that open each block - see the coverage note further
# down for why, with local data, this only ever covers the aud-rewarded
# blocks) are pulled out of every overlay group and drawn as their own
# traces, one per trial, coloured by BLOCK (each block gets its own hue,
# evenly spaced, so several can be told apart at once) with a light
# (earliest) -> dark (latest) shade running across that block's own up-to-5
# trials; dashed = incorrect. A "show task-switch trials" checkbox (unchecked
# by default, so the figure opens without them) gates a dual-knob range
# slider (built further down, next to the ANIMATE_3D time slider) that picks
# the [from, to] window of task-switch trials shown once enabled; by default
# the trial at one edge of that window (position configurable, default "end"
# = most recent) is drawn fully opaque and the rest fade out with distance
# from it - a second checkbox turns that fade off (all shown at full,
# equal opacity). One checkbox per switching block (labelled with the
# transition it represents, e.g. "block 4 (vis→aud)", or just the modality
# for the very first block) sits above the slider - any combination,
# including several non-consecutive blocks, can be ticked at once to pool
# them into the slider's chronological window. With exactly one block
# ticked, "before"/"after" number inputs additionally appear to reveal up to
# MAX_CONTEXT_TRIALS ordinary trials immediately preceding/following that
# block's task-switch trials (independently sized, so the context need not
# be symmetric), drawn as faint grey dotted lines for chronological context.
#
# ANIMATE_3D adds a Play/Pause button (its own row, above the time slider - the
# two used to share a row and the buttons' width ran into the slider) and a
# time slider that sweep every trajectory - the 4 means (with a moving head
# marker) and, when TRIALS_3D is on, each single trial in the overlay - from
# the first bin to the last. The axis ranges are frozen while animating so the
# view does not jump.
#
# Every per-class marker - the moving head (circle) always, and (when
# EVENT_MARKERS_3D is on) the start/stimulus-onset markers (square/diamond) -
# is tied to its own MEAN trace rather than shown unconditionally: visible
# only while that class's MEAN trace is legend-selected, and for start/onset
# specifically, only once the animation has actually drawn past that marker's
# time bin (so, in the default fully-drawn view, always; while animating,
# each pops in as the line reaches it). Deselecting a class in the legend
# hides its head/start/onset markers immediately, along with its trace. This
# is wired up client-side (see the marker-sync post_script below) since
# Plotly's legend and its animation system don't otherwise know about each
# other.
#
# Click any trace (line or marker) to isolate it: it snaps to full opacity
# and every other trace fades towards transparent; click the same trace again
# to bring everything back. This is a separate, generic post_script (applies
# to the whole figure, not any one feature above) that snapshots each trace's
# current opacity right before dimming and restores exactly that on
# toggle-off, so it layers on top of whatever the task-switch fade slider or
# marker-sync happen to have set rather than clobbering them.
import colorsys
import json
import subprocess
import webbrowser

import plotly.graph_objects as go

TRIALS_3D = "outcome"       # "none" | "combo" | "outcome"
TRIALS_3D_COMBOS = None     # None = all conditions; else a set of (stim, context)
ANIMATE_3D = True           # add Play/Pause + a time slider; grows means and single trials
EVENT_MARKERS_3D = True     # draw the trial-start / stimulus-onset markers (either way
                            # they are single legend entries you can click on/off)
MARK_REWARD_SCHEDULED = True  # pull the task-switch (is_reward_scheduled / instruction)
                             # trials out of each overlay group and draw them as thin
                             # purple lines - deep purple correct, light purple incorrect -
                             # with their own legend entry. These are the 5 instruction
                             # trials that open every aud-rewarded block (3 blocks -> 15).
MAX_CONTEXT_TRIALS = 15     # cap on the "trials before"/"trials after" context traces
                             # precomputed per block for the task-switch widget - the
                             # widget's before/after inputs can't reveal more than this
                             # (silently clamped near the start/end of the session)

pca3 = PCA(n_components=3)
pca3.fit(flat_activity[flat_fit_mask])   # fit on non-scheduled trials, project everyone
scores3 = pca3.transform(flat_activity).reshape(n_trials_pca, n_bins_pca, 3)
# 2 d.p. is well below the noise floor of these scores and keeps the animated
# HTML (which repeats every trial's coordinates in every frame) a sane size;
# float64 so the rounded values serialise as short decimals ("1.23"), not the
# long float32 repr
scores3 = np.round(scores3.astype(np.float64), 2)
var3 = 100 * pca3.explained_variance_ratio_

correct_per_trial = aud_trials["is_correct"].to_numpy(dtype=bool)
incorrect_per_trial = aud_trials["is_incorrect"].to_numpy(dtype=bool)
# sched_per_trial was already computed in the processing-step cell above (the
# trials excluded from every fit, here only ever projected)

combo_style = {
    ("sound1", "aud"): ("sound1 / aud-rewarded", "#1f5fa8"),
    ("sound1", "vis"): ("sound1 / vis-rewarded", "#7fb3e6"),
    ("sound2", "aud"): ("sound2 / aud-rewarded", "#c74a1e"),
    ("sound2", "vis"): ("sound2 / vis-rewarded", "#f0a878"),
}

def _paths_xyz(paths, k=None):
    """Flatten an (n, bins, 3) array to x/y/z lists with NaN gaps between paths.

    With ``k`` set, each path is truncated to its first ``k`` samples (used to
    grow the single-trial lines frame by frame).
    """
    xs, ys, zs = [], [], []
    for p in paths:
        q = p if k is None else p[:k]
        xs.extend(q[:, 0]); xs.append(np.nan)
        ys.extend(q[:, 1]); ys.append(np.nan)
        zs.extend(q[:, 2]); zs.append(np.nan)
    return xs, ys, zs

if TRIALS_3D_COMBOS is None:
    trial_subset = np.ones(len(stim_name_per_trial), dtype=bool)
else:
    trial_subset = np.zeros(len(stim_name_per_trial), dtype=bool)
    for _s, _c in TRIALS_3D_COMBOS:
        trial_subset |= (stim_name_per_trial == _s) & (context_per_trial == _c)

fig3d = go.Figure()

# single-trial overlay first, so the mean trajectories sit on top of it.
# anim_trial_groups keeps (trace index, its (n, bins, 3) paths) so the frames
# below can regrow each single trial along with the means.
anim_trial_groups = []

def _add_overlay(sel, color, label, base_name, showlegend, opacity, width):
    """Add the single-trial trace for one group. With MARK_REWARD_SCHEDULED,
    reward-scheduled (task-switch) trials are excluded here - they get their
    own individually-coloured, order-revealing traces added below instead."""
    if MARK_REWARD_SCHEDULED:
        sel = sel & ~sched_per_trial
    if sel.sum() == 0:
        return
    paths = scores3[sel]
    gx, gy, gz = _paths_xyz(paths)
    anim_trial_groups.append((len(fig3d.data), paths))
    fig3d.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode="lines", opacity=opacity,
        line=dict(color=color, width=width),
        legendgroup=label, showlegend=showlegend, hoverinfo="none",
        visible="legendonly",   # only the MEAN traces are selected by default
        name=f"{base_name} (n={int(sel.sum())})" if showlegend else None))

if TRIALS_3D == "combo":
    for (stim_name, context), (label, color) in combo_style.items():
        mask = ((stim_name_per_trial == stim_name)
                & (context_per_trial == context) & trial_subset)
        _add_overlay(mask, color, label, label, showlegend=False, opacity=0.12, width=1)
elif TRIALS_3D == "outcome":
    # correct / incorrect single trials, kept separate for each of the 4
    # stimulus x context classes; entries sit next to their class mean in the
    # legend and toggle one at a time.
    for (stim_name, context), (label, _) in combo_style.items():
        combo_mask = (stim_name_per_trial == stim_name) & (context_per_trial == context)
        for outcome, sel_all, ocolor in (
                ("correct", correct_per_trial, "#2ca02c"),
                ("incorrect", incorrect_per_trial, "#d62728")):
            sel = combo_mask & sel_all & trial_subset
            _add_overlay(sel, ocolor, label, f"{label} - {outcome}",
                         showlegend=True, opacity=0.3, width=1.5)

anim_mean_idx, mean_paths, mean_colors, mean_labels = [], [], [], []
mean_trace_idx = {}   # label -> MEAN trace index, for the marker-sync script below
for (stim_name, context), (label, color) in combo_style.items():
    mask = (stim_name_per_trial == stim_name) & (context_per_trial == context)
    if mask.sum() < 2:
        continue
    m = scores3[mask].mean(axis=0)
    mean_trace_idx[label] = len(fig3d.data)
    anim_mean_idx.append(len(fig3d.data))
    mean_paths.append(m)
    mean_colors.append(color)
    mean_labels.append(label)
    fig3d.add_trace(go.Scatter3d(
        x=m[:, 0], y=m[:, 1], z=m[:, 2], mode="lines+markers",
        line=dict(color=color, width=6), marker=dict(size=3, color=color),
        legendgroup=label, name=f"{label} - MEAN (n={int(mask.sum())})",
        customdata=np.arange(n_bins_pca),
        hovertemplate=("time bin %{customdata}<br>PC1 %{x:.2f}<br>"
                       "PC2 %{y:.2f}<br>PC3 %{z:.2f}<extra></extra>")))

# Reward-scheduled ("task-switch") trials each get their own trace, grouped
# and coloured by BLOCK rather than by one global chronological gradient: each
# block that has any task-switch trial gets its own hue (evenly spaced, so
# blocks stay visually distinct even when several non-consecutive ones are
# shown together), and within a block the shade runs light -> dark as its
# up-to-5 instruction trials proceed (so the block's own progression is still
# readable). Correctness is orthogonal to colour: dashed = incorrect.
#
# NOTE ON COVERAGE: only aud-rewarded blocks currently have any task-switch
# trace to draw. is_reward_scheduled *is* set correctly for every block in
# */trials.csv (5 trials/block, vis-rewarded blocks' on stim_name="vis1"), but
# local DATA/ only has AUD-unit spikes for sound1/sound2 trials - vis1's own
# spike export (vis1_spike_times.csv) covers a different region entirely
# (VISli, 18 units, zero overlap with the 144 AUD units used here) - see the
# data-availability memory. So the vis-rewarded blocks' instruction trials
# have no AUD population activity to project, and can't be drawn until AUD
# spikes for vis1 trials are exported from the capsule.
def _block_shades(hue_frac, n, light_hi=0.78, light_lo=0.32, sat=0.65):
    """n colours for one block, same hue, light (earliest trial) to dark (latest)."""
    shades = []
    for i in range(n):
        light = light_hi if n <= 1 else light_hi - (light_hi - light_lo) * i / (n - 1)
        r, g, b = colorsys.hls_to_rgb(hue_frac, light, sat)
        shades.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")
    return shades

sched_trace_idx, sched_trial_pts = [], np.empty((0, 3))
sched_block_all = []           # block number per entry of sched_trace_idx, same order
context_trace_idx = {}         # block -> {"before": [trace idx,...], "after": [...]}
context_trial_pts = np.empty((0, 3))
block_transition_label = {}    # block -> "aud" / "vis" / "aud→vis" / "vis→aud"
block_color_swatch = {}        # block -> representative CSS colour (for the block checkbox)
if MARK_REWARD_SCHEDULED:
    sched_idx_all = np.flatnonzero(sched_per_trial & trial_subset)
    block_of_trial = aud_trials["block_index"].to_numpy()
    modality_of_block = aud_trials.groupby("block_index")["rewarded_modality"].first()
    block_values = sorted(set(int(block_of_trial[p]) for p in sched_idx_all))
    n_sched = len(sched_idx_all)
    if n_sched > 0:
        sched_trial_pts = scores3[sched_idx_all].reshape(-1, 3)
        n_trials_total = len(aud_trials)
        context_pts_parts = []
        for bi, block in enumerate(block_values):
            block_positions = sorted(
                int(p) for p in sched_idx_all if int(block_of_trial[p]) == block)
            cur_mod = modality_of_block.get(block, "?")
            prev_mod = modality_of_block.get(block - 1) if block > modality_of_block.index.min() else None
            block_transition_label[block] = (
                cur_mod if prev_mod is None or prev_mod == cur_mod else f"{prev_mod}→{cur_mod}")

            hue = ((30 + 360 * bi / len(block_values)) % 360) / 360
            shades = _block_shades(hue, len(block_positions))
            _sr, _sg, _sb = colorsys.hls_to_rgb(hue, 0.55, 0.65)   # one representative
            block_color_swatch[block] = f"rgb({int(_sr * 255)},{int(_sg * 255)},{int(_sb * 255)})"
            for order, trial_pos in enumerate(block_positions):
                is_correct = bool(correct_per_trial[trial_pos])
                gx, gy, gz = _paths_xyz([scores3[trial_pos]])
                sched_trace_idx.append(len(fig3d.data))
                sched_block_all.append(block)
                anim_trial_groups.append((len(fig3d.data), scores3[trial_pos][None]))
                fig3d.add_trace(go.Scatter3d(
                    x=gx, y=gy, z=gz, mode="lines",
                    line=dict(color=shades[order], width=3, dash="solid" if is_correct else "dot"),
                    legendgroup="task-switch", showlegend=True, hoverinfo="none",
                    name=(f"task-switch block {block} ({block_transition_label[block]}) "
                         f"trial {order + 1}/{len(block_positions)} "
                         f"({'correct' if is_correct else 'incorrect'})")))

            # Context trials: up to MAX_CONTEXT_TRIALS ordinary trials immediately
            # before/after this block's task-switch trials (by chronological
            # trial position - aud_trials/scores3 are already sorted by trial
            # id), added as their own hidden traces so the widget can reveal an
            # asymmetric number of them on either side once this block is
            # selected. Assumes the block's task-switch trials are contiguous
            # (they are the 5 instruction trials that *open* the block).
            p_min, p_max = block_positions[0], block_positions[-1]
            before_positions = list(range(max(0, p_min - MAX_CONTEXT_TRIALS), p_min))
            after_positions = list(range(p_max + 1, min(n_trials_total, p_max + 1 + MAX_CONTEXT_TRIALS)))
            before_idx, after_idx = [], []
            for pos in before_positions + after_positions:
                gx, gy, gz = _paths_xyz([scores3[pos]])
                anim_trial_groups.append((len(fig3d.data), scores3[pos][None]))
                (before_idx if pos in before_positions else after_idx).append(len(fig3d.data))
                fig3d.add_trace(go.Scatter3d(
                    x=gx, y=gy, z=gz, mode="lines",
                    line=dict(color="#7a7a7a", width=2, dash="dot"),
                    legendgroup="task-switch-context", showlegend=False, hoverinfo="none",
                    visible=False,
                    name=f"context trial (block {block}, position {pos})"))
            context_trace_idx[block] = {"before": before_idx, "after": after_idx}
            if before_positions or after_positions:
                context_pts_parts.append(scores3[before_positions + after_positions].reshape(-1, 3))
        if context_pts_parts:
            context_trial_pts = np.concatenate(context_pts_parts, axis=0)

# Per-class start / stimulus-onset markers (square / diamond) and the moving
# "current time" head marker (circle) - one trace per class per marker type,
# rather than one trace shared across all 4 classes, so each marker's
# visibility can be tied to its OWN class: on only while that class's MEAN
# trace is legend-selected (the head marker), and additionally only once the
# animated trace has actually drawn past that point in time (start/onset).
# Neither condition is something Plotly links up on its own (legend clicks
# and animation are independent of each other), so both are evaluated
# client-side by MARKER_SYNC_JS, appended to _post_scripts further down.
# EVENT_MARKERS_3D=False skips start/onset entirely; the head marker is
# always added (it's the animation's current-position indicator, not really
# an optional annotation) but still only shown for a selected class.
marker_trace_idx = {}   # label -> {"start": idx, "onset": idx, "head": idx}
last_bin = n_bins_pca - 1
for label, color, path in zip(mean_labels, mean_colors, mean_paths):
    marker_trace_idx[label] = {}
    if EVENT_MARKERS_3D:
        for _event, _bin, _sym, _sz in (("start", 0, "square", 7),
                                        ("onset", onset_bin, "diamond", 8)):
            marker_trace_idx[label][_event] = len(fig3d.data)
            fig3d.add_trace(go.Scatter3d(
                x=[path[_bin, 0]], y=[path[_bin, 1]], z=[path[_bin, 2]],
                mode="markers", showlegend=False, hoverinfo="none",
                marker=dict(size=_sz, color=color, symbol=_sym,
                            line=dict(color="black", width=1))))
    marker_trace_idx[label]["head"] = len(fig3d.data)
    fig3d.add_trace(go.Scatter3d(
        x=[path[-1, 0]], y=[path[-1, 1]], z=[path[-1, 2]], mode="markers",
        marker=dict(size=6, color=color, symbol="circle",
                    line=dict(color="black", width=1)),
        showlegend=False, hoverinfo="none"))
heads_idx = [marker_trace_idx[label]["head"] for label in mean_labels]

_overlay_note = {"none": "means only",
                 "combo": "single trials coloured by condition",
                 "outcome": "single trials per class, green = correct / red = incorrect"}[TRIALS_3D]

# Fixed axis ranges (never autoscaled): span everything that gets drawn, plus a
# small margin. The box shape is pinned with aspectmode="manual" + an explicit
# aspectratio so nothing about it depends on the per-frame data; the same scene
# ranges are repeated in every animation frame's layout, because a plotly 3D
# animation otherwise re-fits the scene to each frame.
_pts = np.concatenate(mean_paths, axis=0)
if TRIALS_3D != "none":
    _pts = np.vstack([_pts, scores3[trial_subset].reshape(-1, 3)])
if len(sched_trial_pts):
    _pts = np.vstack([_pts, sched_trial_pts])
if len(context_trial_pts):
    _pts = np.vstack([_pts, context_trial_pts])
_lo, _hi = _pts.min(axis=0), _pts.max(axis=0)
_pad = 0.06 * (_hi - _lo)
_ax_lo, _ax_hi = _lo - _pad, _hi + _pad
_span = _ax_hi - _ax_lo
_aspect = dict(zip("xyz", (_span / _span.max()).tolist()))

def _fixed_scene(with_titles):
    s = dict(aspectmode="manual", aspectratio=_aspect)
    for i, ax in enumerate(("xaxis", "yaxis", "zaxis")):
        s[ax] = dict(range=[float(_ax_lo[i]), float(_ax_hi[i])], autorange=False)
        if with_titles:
            s[ax]["title"] = f"PC{i + 1} ({var3[i]:.0f}% var)"
    return s

_scene = _fixed_scene(with_titles=True)

if ANIMATE_3D:
    _frame_layout = go.Layout(scene=_fixed_scene(with_titles=False))
    frames = []
    for k in range(2, n_bins_pca + 1):
        fdata = [go.Scatter3d(x=m[:k, 0], y=m[:k, 1], z=m[:k, 2]) for m in mean_paths]
        ftraces = list(anim_mean_idx)
        for idx, paths in anim_trial_groups:            # grow the single trials too
            gx, gy, gz = _paths_xyz(paths, k)
            fdata.append(go.Scatter3d(x=gx, y=gy, z=gz))
            ftraces.append(idx)
        for head_idx, m in zip(heads_idx, mean_paths):   # move each class's own head marker
            fdata.append(go.Scatter3d(x=[m[k - 1, 0]], y=[m[k - 1, 1]], z=[m[k - 1, 2]]))
            ftraces.append(head_idx)
        frames.append(go.Frame(name=str(k), data=fdata, traces=ftraces,
                               layout=_frame_layout))
    fig3d.frames = frames

    _play = dict(frame=dict(duration=90, redraw=True),
                 transition=dict(duration=0), mode="immediate", fromcurrent=False)
    _pause = dict(frame=dict(duration=0, redraw=False),
                  transition=dict(duration=0), mode="immediate")
    # Play/Pause sits on its own row above the time slider (rather than to its
    # left on the same row) - sharing a row left the two buttons' rendered
    # width overflowing into the slider's start, so the buttons were "in the
    # way" of dragging it. Stacking removes any chance of that regardless of
    # exact button width; margin.b (below) is widened to fit both rows.
    _scene_menu = dict(type="buttons", direction="left", showactive=False,
                       x=0.02, y=0.14, xanchor="left", yanchor="bottom", pad=dict(r=6, t=6),
                       buttons=[dict(label="▶ Play", method="animate", args=[None, _play]),
                                dict(label="▮▮ Pause", method="animate",
                                     args=[[None], _pause])])
    _slider = dict(active=len(frames) - 1, x=0.02, y=0.02, len=0.94,
                   xanchor="left", yanchor="bottom", pad=dict(t=4, b=4),
                   currentvalue=dict(prefix="t = ", suffix=" s", xanchor="right"),
                   steps=[dict(method="animate", label=f"{bin_centers[int(f.name) - 1]:+.2f}",
                               args=[[f.name], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True),
                                                   transition=dict(duration=0))])
                          for f in frames])
    fig3d.update_layout(updatemenus=[_scene_menu])
    _overlay_note += "; press Play or drag the time slider to sweep time"

_sliders = [_slider] if ANIMATE_3D else []
_post_scripts = []

# Marker-sync script: keeps every start/onset marker's visibility equal to
# (its class's MEAN trace is legend-selected) AND (the animation has drawn
# past that marker's time bin), and keeps the head marker's visibility equal
# to just the first condition (it has no fixed bin - it's wherever the
# animation currently is, so there's nothing for it to "pass"). Neither
# condition is something Plotly tracks together on its own - legend clicks
# fire "plotly_restyle" and have no idea what time the animation is at, while
# frame changes ("plotly_animatingframe") have no idea which classes are
# currently selected - so this listens to both and recomputes every marker's
# visibility from scratch each time.
if marker_trace_idx:
    _marker_sync_js = r"""
(function() {
    var gd = document.getElementById('{plot_id}');
    var meanTraceIdx = __MEAN_TRACE_IDX__;        // {label: trace idx}
    var markerTraceIdx = __MARKER_TRACE_IDX__;    // {label: {start, onset, head: trace idx}}
    var onsetBin = __ONSET_BIN__;
    var lastBin = __LAST_BIN__;
    var currentBin = lastBin;   // default (non-animating) view: the full trace is drawn
    var syncing = false;        // guards against our own restyle re-triggering itself

    function hasPassed(bin) {
        return currentBin > bin;
    }

    function isSelected(traceIdx) {
        var v = gd.data[traceIdx].visible;
        return v === true || v === undefined;   // Plotly's own default is true
    }

    function syncMarkers() {
        if (syncing) { return; }
        var idxList = [], targets = [];
        Object.keys(meanTraceIdx).forEach(function(label) {
            var selected = isSelected(meanTraceIdx[label]);
            var markers = markerTraceIdx[label] || {};
            // null bin (the head marker) means "no time gating - selected is enough"
            [['start', 0], ['onset', onsetBin], ['head', null]].forEach(function(pair) {
                var mIdx = markers[pair[0]];
                if (mIdx === undefined) { return; }
                var want = selected && (pair[1] === null || hasPassed(pair[1]));
                if (gd.data[mIdx].visible !== want) { idxList.push(mIdx); targets.push(want); }
            });
        });
        if (idxList.length) {
            syncing = true;
            Plotly.restyle(gd, {visible: targets}, idxList).then(function() { syncing = false; },
                                                                 function() { syncing = false; });
        }
    }

    gd.on('plotly_restyle', syncMarkers);
    gd.on('plotly_animatingframe', function(frame) {
        currentBin = parseInt(frame.name, 10) - 1;
        syncMarkers();
    });
    syncMarkers();
})();
"""
    _post_scripts.append(_marker_sync_js
                          .replace("__MEAN_TRACE_IDX__", json.dumps(mean_trace_idx))
                          .replace("__MARKER_TRACE_IDX__", json.dumps(marker_trace_idx))
                          .replace("__ONSET_BIN__", str(onset_bin))
                          .replace("__LAST_BIN__", str(last_bin)))

# Click-to-highlight: click any trace (line or marker) to make it fully
# opaque and fade every other trace towards transparent; click the same
# trace again to restore everything. This is a generic overlay on top of
# whatever opacity the other systems above (task-switch fade slider,
# marker-sync) currently have each trace set to - it snapshots that state
# right before dimming and restores exactly that snapshot on toggle-off,
# rather than some fixed default, so it composes with them instead of
# fighting over a trace's opacity. hoverinfo="none" on the line/marker traces
# above (rather than "skip") is required for this: Plotly's gl3d click
# detection reuses its hover point-picking, and "skip" excludes a trace from
# that entirely, so it would never appear as click event's target.
_click_highlight_js = r"""
(function() {
    var gd = document.getElementById('{plot_id}');
    var highlighted = null;     // currently-highlighted trace index, or null
    var baseOpacity = null;     // opacity snapshot from just before highlighting began
    var DIM_OPACITY = 0.08;

    function currentOpacity() {
        return gd.data.map(function(t) { return t.opacity === undefined ? 1 : t.opacity; });
    }

    gd.on('plotly_click', function(evt) {
        if (!evt || !evt.points || !evt.points.length) { return; }
        var clicked = evt.points[0].curveNumber;
        if (clicked === highlighted) {
            Plotly.restyle(gd, {opacity: baseOpacity});
            highlighted = null;
            baseOpacity = null;
            return;
        }
        if (highlighted === null) { baseOpacity = currentOpacity(); }
        highlighted = clicked;
        Plotly.restyle(gd, {opacity: gd.data.map(function(_, i) {
            return i === clicked ? 1 : DIM_OPACITY;
        })});
    });
})();
"""
_post_scripts.append(_click_highlight_js)

# Task-switch WINDOW slider: a single dual-knob range slider (built from plain
# HTML/CSS/JS in post_script, since Plotly's own sliders are single-handle
# only) picks the chronological range of task-switch trials shown. Both knobs
# sit on the same track - drag either one to resize the window, or grab the
# highlighted bar between them to roll the whole window (fixed width) left or
# right. Alongside it, a checkbox (checked by default) fades every task-switch
# trace in the window towards transparent by its distance from one edge, so
# only the trial at that edge (or the middle, see the anchor radios) stays
# fully opaque - the other traces just give chronological context. Unchecking
# it shows every trial in the window at full, equal opacity instead. Dragging
# or toggling either control restyles the task-switch traces directly,
# independent of the ANIMATE_3D time slider above (that one only ever touches
# x/y/z; this one only ever touches "visible"/"opacity").
if sched_trace_idx:
    _widget_js = r"""
(function() {
    // post_script runs inside a .then(function(){...}) with no arguments bound,
    // so the graph div is NOT available as a free variable - it has to be looked
    // up by id. '{plot_id}' is a literal placeholder that plotly.py substitutes
    // with the div's actual id before this script is ever sent to the browser.
    var gd = document.getElementById('{plot_id}');
    var schedIdx = __SCHED_IDX__;              // trace id per task-switch trial, chronological
    var schedBlockAll = __SCHED_BLOCK__;       // block number per entry of schedIdx, same order
    var blockValues = __BLOCK_VALUES__;        // distinct block numbers, e.g. [0, 2, 4]
    var blockLabel = __BLOCK_LABEL__;          // {block: "aud" | "vis" | "vis→aud" | ...}
    var blockColor = __BLOCK_COLOR__;          // {block: css colour} - matches the block's trace hue
    var contextTraceIdx = __CONTEXT_TRACE_IDX__;  // {block: {before: [trace idx,...], after: [...]}}
    var nSchedAll = schedIdx.length;
    var nSched = nSchedAll;                    // trial count in the CURRENT block filter
    var lo = 0, hi = nSched - 1;                // window bounds, as local indices into that filter
    var selectedBlocks = {};                    // {block (string): true} - all blocks start checked
    blockValues.forEach(function(b) { selectedBlocks[b] = true; });
    var showEnabled = false;   // task-switch trials are off until this is checked
    var fadeEnabled = true;
    var anchorMode = 'end';   // 'begin' | 'center' | 'end' - which edge stays opaque
    var MIN_OPACITY = 0.08;
    var MIN_HANDLE_GAP_PX = 14;   // keep the two knobs apart so neither hides the other

    var container = document.createElement('div');
    container.id = 'task-switch-window-slider';
    container.innerHTML =
        '<style>' +
        '#task-switch-window-slider{font-family:sans-serif;font-size:13px;max-width:900px;margin:8px auto 24px;padding:0 24px;}' +
        '#task-switch-window-slider .ts-master{display:block;margin-bottom:10px;color:#333;cursor:pointer;}' +
        '#task-switch-window-slider .ts-body{transition:opacity .15s;}' +
        '#task-switch-window-slider .ts-body.ts-disabled{opacity:0.5;}' +
        '#task-switch-window-slider .ts-block-row{margin-bottom:10px;display:flex;' +
            'align-items:center;gap:14px;flex-wrap:wrap;color:#333;}' +
        '#task-switch-window-slider .ts-block-row>span{color:#777;}' +
        '#task-switch-window-slider .ts-block-cb{display:inline-flex;align-items:center;' +
            'gap:5px;cursor:pointer;}' +
        '#task-switch-window-slider .ts-swatch{display:inline-block;width:11px;height:11px;' +
            'border-radius:50%;border:1px solid rgba(0,0,0,0.3);}' +
        '#task-switch-window-slider .ts-label{margin-bottom:8px;color:#333;}' +
        '#task-switch-window-slider .ts-track-wrap{position:relative;height:28px;}' +
        '#task-switch-window-slider .ts-track{position:absolute;top:12px;left:0;right:0;height:4px;background:#ddd;border-radius:2px;}' +
        '#task-switch-window-slider .ts-range{position:absolute;top:12px;height:4px;background:#7fb3e6;border-radius:2px;cursor:grab;}' +
        '#task-switch-window-slider .ts-range:active{cursor:grabbing;}' +
        '#task-switch-window-slider .ts-handle{position:absolute;top:5px;width:18px;height:18px;margin-left:-9px;' +
            'border-radius:50%;background:#1f5fa8;border:2px solid white;box-shadow:0 0 2px rgba(0,0,0,0.5);cursor:pointer;touch-action:none;}' +
        '#task-switch-window-slider .ts-controls{margin-top:10px;display:flex;align-items:center;' +
            'gap:18px;flex-wrap:wrap;color:#333;}' +
        '#task-switch-window-slider .ts-controls label{cursor:pointer;margin-right:4px;}' +
        '#task-switch-window-slider .ts-anchor-group label{margin-left:4px;margin-right:10px;}' +
        '#task-switch-window-slider .ts-context-row{margin-top:10px;display:flex;align-items:center;' +
            'gap:14px;flex-wrap:wrap;color:#333;}' +
        '#task-switch-window-slider .ts-context-row.ts-hidden{display:none;}' +
        '#task-switch-window-slider .ts-context-row input[type=number]{width:52px;}' +
        '#task-switch-window-slider .ts-context-note{color:#777;font-size:12px;}' +
        '</style>' +
        '<label class="ts-master"><input type="checkbox" id="ts-show-toggle"> ' +
            'show task-switch (block-switching) trials</label>' +
        '<div class="ts-body ts-disabled" id="ts-body">' +
            '<div class="ts-block-row" id="ts-block-row"><span>switching blocks:</span></div>' +
            '<div class="ts-label" id="ts-label"></div>' +
            '<div class="ts-track-wrap" id="ts-track-wrap">' +
                '<div class="ts-track"></div>' +
                '<div class="ts-range" id="ts-range"></div>' +
                '<div class="ts-handle" id="ts-handle-lo"></div>' +
                '<div class="ts-handle" id="ts-handle-hi"></div>' +
            '</div>' +
            '<div class="ts-controls">' +
                '<label><input type="checkbox" id="ts-fade-toggle" checked> fade older task-switch trials</label>' +
                '<span class="ts-anchor-group">highlight the' +
                    '<label><input type="radio" name="ts-anchor" value="begin"> beginning</label>' +
                    '<label><input type="radio" name="ts-anchor" value="center"> center</label>' +
                    '<label><input type="radio" name="ts-anchor" value="end" checked> end</label>' +
                    'of the window</span>' +
            '</div>' +
            '<div class="ts-context-row ts-hidden" id="ts-context-row">' +
                '<span>surrounding trials -</span>' +
                '<label>before <input type="number" id="ts-before-input" min="0" max="0" value="0"></label>' +
                '<label>after <input type="number" id="ts-after-input" min="0" max="0" value="0"></label>' +
                '<span class="ts-context-note" id="ts-context-note"></span>' +
            '</div>' +
        '</div>';
    gd.parentNode.insertBefore(container, gd.nextSibling);

    var showToggle = container.querySelector('#ts-show-toggle');
    var body = container.querySelector('#ts-body');
    var blockRow = container.querySelector('#ts-block-row');
    var track = container.querySelector('#ts-track-wrap');
    var rangeEl = container.querySelector('#ts-range');
    var handleLo = container.querySelector('#ts-handle-lo');
    var handleHi = container.querySelector('#ts-handle-hi');
    var label = container.querySelector('#ts-label');
    var fadeToggle = container.querySelector('#ts-fade-toggle');
    var anchorRadios = container.querySelectorAll('input[name="ts-anchor"]');
    var contextRow = container.querySelector('#ts-context-row');
    var beforeInput = container.querySelector('#ts-before-input');
    var afterInput = container.querySelector('#ts-after-input');
    var contextNote = container.querySelector('#ts-context-note');

    blockValues.forEach(function(b) {
        var lbl = document.createElement('label');
        lbl.className = 'ts-block-cb';
        lbl.innerHTML = '<input type="checkbox" checked value="' + b + '"> ' +
            '<span class="ts-swatch" style="background:' + blockColor[b] + '"></span> ' +
            'block ' + b + ' (' + blockLabel[b] + ')';
        blockRow.appendChild(lbl);
    });
    var blockCheckboxes = container.querySelectorAll('.ts-block-cb input[type=checkbox]');

    function singleSelectedBlock() {
        var chosen = Object.keys(selectedBlocks).filter(function(b) { return selectedBlocks[b]; });
        return chosen.length === 1 ? chosen[0] : null;
    }

    function refresh() { applyWindow(); render(); }

    showToggle.addEventListener('change', function() {
        showEnabled = showToggle.checked;
        body.classList.toggle('ts-disabled', !showEnabled);
        refresh();
    });
    for (var bc = 0; bc < blockCheckboxes.length; bc++) {
        blockCheckboxes[bc].addEventListener('change', function(e) {
            selectedBlocks[e.target.value] = e.target.checked;
            lo = 0;
            hi = Number.MAX_SAFE_INTEGER;   // reset to the new filter's full range; applyWindow() clamps it
            updateContextControlsForBlock();
            refresh();
        });
    }
    fadeToggle.addEventListener('change', function() {
        fadeEnabled = fadeToggle.checked;
        refresh();
    });
    for (var ri = 0; ri < anchorRadios.length; ri++) {
        anchorRadios[ri].addEventListener('change', function(e) {
            if (e.target.checked) { anchorMode = e.target.value; refresh(); }
        });
    }
    beforeInput.addEventListener('input', applyContextTrials);
    afterInput.addEventListener('input', applyContextTrials);

    function pct(i) { return nSched > 1 ? (100 * i / (nSched - 1)) : 0; }

    function render() {
        // Position the knobs in pixels (not just percent) so that when they land
        // on the same index - e.g. after dragging one all the way onto the other -
        // they can be nudged apart by a minimum gap instead of exactly overlapping.
        // At exact overlap the later-painted knob covers the other completely, so
        // its mousedown/touchstart listener can never fire again - that was the
        // "can't grab the left knob any more" bug.
        var trackWidth = track.getBoundingClientRect().width || 1;
        var loPx = pct(lo) / 100 * trackWidth;
        var hiPx = pct(hi) / 100 * trackWidth;
        if (hiPx - loPx < MIN_HANDLE_GAP_PX) {
            var mid = (loPx + hiPx) / 2;
            loPx = mid - MIN_HANDLE_GAP_PX / 2;
            hiPx = mid + MIN_HANDLE_GAP_PX / 2;
        }
        handleLo.style.left = loPx + 'px';
        handleHi.style.left = hiPx + 'px';
        rangeEl.style.left = loPx + 'px';
        rangeEl.style.right = (trackWidth - hiPx) + 'px';
        var single = singleSelectedBlock();
        var blockNote = single !== null ? ' (block ' + single + ')' : '';
        label.textContent = !showEnabled
            ? 'task-switch trials hidden - check the box above to show them'
            : nSched === 0
            ? 'no switching blocks selected - check at least one above'
            : ('task-switch trials shown: ' + (lo + 1) + '-' + (hi + 1) + ' of ' + nSched +
               blockNote + ' (drag a knob to resize the window, drag the bar to roll it)');
    }

    function opacityAt(i) {
        if (!fadeEnabled) { return 1; }
        var anchor = anchorMode === 'begin' ? lo : anchorMode === 'end' ? hi
            : Math.round((lo + hi) / 2);
        var maxDist = Math.max(anchor - lo, hi - anchor) || 1;
        var dist = Math.abs(i - anchor);
        return MIN_OPACITY + (1 - MIN_OPACITY) * (1 - dist / maxDist);
    }

    function activeIndices() {
        // positions (0..nSchedAll-1) of the task-switch trials whose block is
        // currently checked (possibly several, non-consecutive), kept in
        // their original chronological order
        var out = [];
        for (var j = 0; j < nSchedAll; j++) {
            if (selectedBlocks[schedBlockAll[j]]) { out.push(j); }
        }
        return out;
    }

    function applyWindow() {
        var active = activeIndices();
        nSched = active.length;
        lo = Math.max(0, Math.min(lo, Math.max(nSched - 1, 0)));
        hi = Math.max(lo, Math.min(hi, Math.max(nSched - 1, 0)));

        var visible = new Array(nSchedAll).fill(false);
        var opacity = new Array(nSchedAll).fill(1);
        for (var j = 0; j < nSched; j++) {
            var globalI = active[j];
            var inWindow = showEnabled && j >= lo && j <= hi;
            visible[globalI] = inWindow;
            opacity[globalI] = inWindow ? opacityAt(j) : 1;
        }
        Plotly.restyle(gd, {visible: visible, opacity: opacity}, schedIdx);
        applyContextTrials();
    }

    function updateContextControlsForBlock() {
        var single = singleSelectedBlock();
        contextRow.classList.toggle('ts-hidden', single === null);
        if (single === null) { return; }
        var entry = contextTraceIdx[single] || {before: [], after: []};
        beforeInput.max = entry.before.length;
        afterInput.max = entry.after.length;
        if (Number(beforeInput.value) > entry.before.length) { beforeInput.value = entry.before.length; }
        if (Number(afterInput.value) > entry.after.length) { afterInput.value = entry.after.length; }
        contextNote.textContent = '(up to ' + entry.before.length + ' before, ' +
            entry.after.length + ' after available)';
    }

    function applyContextTrials() {
        // hide every block's context trials first - only meaningful for a
        // single selected block - then reveal the requested count for it,
        // nearest-to-the-block first
        var allContextIdx = [];
        for (var b in contextTraceIdx) {
            allContextIdx = allContextIdx.concat(contextTraceIdx[b].before, contextTraceIdx[b].after);
        }
        if (allContextIdx.length) {
            Plotly.restyle(gd, {visible: allContextIdx.map(function() { return false; })}, allContextIdx);
        }
        var single = singleSelectedBlock();
        if (single === null || !showEnabled) { return; }
        var entry = contextTraceIdx[single];
        if (!entry) { return; }
        var nBefore = Math.max(0, Math.min(parseInt(beforeInput.value, 10) || 0, entry.before.length));
        var nAfter = Math.max(0, Math.min(parseInt(afterInput.value, 10) || 0, entry.after.length));
        // before[] is stored oldest-first (ascending trial position) - the ones
        // closest to the block, immediately preceding it, are its tail
        var toShow = entry.before.slice(entry.before.length - nBefore).concat(entry.after.slice(0, nAfter));
        if (toShow.length) {
            Plotly.restyle(gd, {visible: toShow.map(function() { return true; }),
                                opacity: toShow.map(function() { return 0.55; })}, toShow);
        }
    }

    function xToIndex(clientX) {
        var rect = track.getBoundingClientRect();
        var frac = (clientX - rect.left) / rect.width;
        frac = Math.max(0, Math.min(1, frac));
        return Math.round(frac * (nSched - 1));
    }

    function startDrag(onMove) {
        function onMouseMove(e) {
            if (e.touches) { e.preventDefault(); }
            onMove(e.touches ? e.touches[0].clientX : e.clientX);
        }
        function onEnd() {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onEnd);
            document.removeEventListener('touchmove', onMouseMove);
            document.removeEventListener('touchend', onEnd);
        }
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onEnd);
        document.addEventListener('touchmove', onMouseMove, {passive: false});
        document.addEventListener('touchend', onEnd);
    }

    function bindHandle(el, isLo) {
        function begin(e) {
            e.preventDefault();
            startDrag(function(clientX) {
                var idx = xToIndex(clientX);
                if (isLo) { lo = Math.min(idx, hi); } else { hi = Math.max(idx, lo); }
                refresh();
            });
        }
        el.addEventListener('mousedown', begin);
        el.addEventListener('touchstart', begin);
    }
    bindHandle(handleLo, true);
    bindHandle(handleHi, false);

    function beginPan(e) {
        e.preventDefault();
        var startClientX = e.touches ? e.touches[0].clientX : e.clientX;
        var startLo = lo, width = hi - lo;
        var rect = track.getBoundingClientRect();
        startDrag(function(clientX) {
            var deltaFrac = (clientX - startClientX) / rect.width;
            var deltaIdx = Math.round(deltaFrac * (nSched - 1));
            lo = Math.max(0, Math.min(nSched - 1 - width, startLo + deltaIdx));
            hi = lo + width;
            refresh();
        });
    }
    rangeEl.addEventListener('mousedown', beginPan);
    rangeEl.addEventListener('touchstart', beginPan);

    updateContextControlsForBlock();
    refresh();
})();
"""
    _post_scripts.append(_widget_js
                          .replace("__SCHED_IDX__", json.dumps(sched_trace_idx))
                          .replace("__SCHED_BLOCK__", json.dumps(sched_block_all))
                          .replace("__BLOCK_VALUES__", json.dumps(block_values))
                          .replace("__BLOCK_LABEL__", json.dumps(block_transition_label))
                          .replace("__BLOCK_COLOR__", json.dumps(block_color_swatch))
                          .replace("__CONTEXT_TRACE_IDX__", json.dumps(context_trace_idx)))
    _overlay_note += ("; drag the two-knob slider below the plot to pick the task-switch "
                      "window shown (drag the bar between the knobs to roll it) - tick any "
                      "combination of switching-block checkboxes to pool them (before/after "
                      "context trials become available when exactly one is ticked), and the "
                      "checkbox/radios below the slider fade all but one edge of the window")

if _sliders:
    fig3d.update_layout(sliders=_sliders)

fig3d.update_layout(
    title=("PC1-PC3 trajectories, auditory cortex (AUDp/AUDv/AUDpo)<br>"
           f"<sup>mean per stimulus x context; {_overlay_note}; square = trial "
           "start, diamond = stimulus onset (each shown only once reached and "
           "while its class is selected); time runs along each line</sup>"),
    scene=_scene,
    legend=dict(itemsizing="constant", groupclick="toggleitem"),
    # b=90 (rather than 0) reserves a dedicated strip below the 3D scene for
    # the Play/Pause row + time slider row, instead of letting them float as
    # an overlay on top of the scene itself; height grows by the same amount
    # so the scene stays the same size as before.
    width=900, height=840, margin=dict(l=0, r=0, t=70, b=90))

_out_html = Path(__file__).resolve().parent / "pca_3d_pc123.html"
_html_kwargs = dict(div_id="pca3d-fig", post_script=_post_scripts or None)
# The task-switch window slider is plain HTML/JS injected via post_script, and
# only the browser page built by write_html actually runs it. VS Code's inline
# plot below (fig3d.show(), "vscode"/plotly_mimetype renderer) ships the figure
# as raw JSON to VS Code's own viewer and ignores post_script and div_id
# entirely - so the slider never appears there; it has to be opened as a file.
#
# Not using write_html's own auto_open=True here: that goes through Python's
# webbrowser module, which under WSL has no Linux browser to launch and no
# bridge to the Windows one - it silently does nothing. explorer.exe opens the
# file with the Windows default browser instead (the usual WSL-to-Windows GUI
# bridge); wslpath -w converts the WSL path to the Windows path it needs.
def _open_in_browser(path):
    is_wsl = "microsoft" in Path("/proc/version").read_text().lower()
    if is_wsl:
        try:
            win_path = subprocess.run(["wslpath", "-w", str(path)], capture_output=True,
                                      text=True, check=True).stdout.strip()
            subprocess.run(["explorer.exe", win_path])
            return
        except Exception as exc:
            print(f"could not auto-open via explorer.exe ({exc}); open {path} manually")
            return
    webbrowser.open(path.as_uri())

fig3d.write_html(_out_html, include_plotlyjs="cdn", auto_open=False, **_html_kwargs)
print(f"wrote interactive plot with task-switch slider -> {_out_html}")
print("(opening it in your browser now; VS Code's inline plot below has no "
      "slider - open the file above to use it)")
_open_in_browser(_out_html)
fig3d.show(**_html_kwargs)
# %%

