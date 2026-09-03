# %%
from pathlib import Path

import pandas as pd

# Path to import data from
DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
UNITS_FILE = DATA_DIR / "good_units.csv"

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
aud_ids = set(good_units.loc[good_units["structure"].str.contains("AUD", na=False), "id"])

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
# Smooth in time, z-score each unit (so no single high-firing unit dominates),
# then run PCA on all (trial, time bin) rows pooled together.
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

SMOOTHING_SIGMA_BINS = 1.5
N_PCS = 3

smoothed = gaussian_filter1d(tensor, sigma=SMOOTHING_SIGMA_BINS, axis=1, mode="nearest")
unit_means = smoothed.mean(axis=(0, 1))
unit_stds = smoothed.std(axis=(0, 1))
active_units = unit_stds > 0   # drop units that never fire
standardised = (smoothed[:, :, active_units] - unit_means[active_units]) / unit_stds[active_units]

n_trials_pca, n_bins_pca, n_units_pca = standardised.shape
flat_activity = standardised.reshape(n_trials_pca * n_bins_pca, n_units_pca)
pca = PCA(n_components=N_PCS)
scores = pca.fit_transform(flat_activity).reshape(n_trials_pca, n_bins_pca, N_PCS)

print(f"PC1-PC{N_PCS} explain {100 * pca.explained_variance_ratio_.sum():.0f}% of variance "
      f"({', '.join(f'{v:.0%}' for v in pca.explained_variance_ratio_)})")

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
scores_grid = (pca_grid.fit_transform(flat_activity)
               .reshape(n_trials_pca, n_bins_pca, N_PCS_GRID))

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
scores_mat = (pca_mat.fit_transform(flat_activity)
              .reshape(n_trials_pca, n_bins_pca, N_PC_MATRIX))
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
# With MARK_REWARD_SCHEDULED, the task-switch trials (is_reward_scheduled: the
# 5 instruction trials that open each aud-rewarded block, 3 blocks -> 15, all
# sound1) are pulled out of each overlay group into thin purple lines - deep
# purple correct, light purple incorrect - each its own legend entry.
#
# ANIMATE_3D adds a Play/Pause button and a time slider that sweep every
# trajectory - the 4 means (with a moving head marker) and, when TRIALS_3D is
# on, each single trial in the overlay - from the first bin to the last. The
# axis ranges are frozen while animating so the view does not jump.
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

pca3 = PCA(n_components=3)
scores3 = pca3.fit_transform(flat_activity).reshape(n_trials_pca, n_bins_pca, 3)
# 2 d.p. is well below the noise floor of these scores and keeps the animated
# HTML (which repeats every trial's coordinates in every frame) a sane size;
# float64 so the rounded values serialise as short decimals ("1.23"), not the
# long float32 repr
scores3 = np.round(scores3.astype(np.float64), 2)
var3 = 100 * pca3.explained_variance_ratio_

correct_per_trial = aud_trials["is_correct"].to_numpy(dtype=bool)
incorrect_per_trial = aud_trials["is_incorrect"].to_numpy(dtype=bool)
sched_per_trial = aud_trials["is_reward_scheduled"].to_numpy(dtype=bool)

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

def _add_overlay(sel, color, label, base_name, showlegend, opacity, width, sched_color):
    """Add the single-trial trace(s) for one group. With MARK_REWARD_SCHEDULED
    the group is split: ordinary trials keep the group colour, while the
    task-switch (is_reward_scheduled) trials are pulled out into a thin
    ``sched_color`` (purple) line with its own legend entry."""
    if MARK_REWARD_SCHEDULED:
        parts = [(sel & ~sched_per_trial, color, False, showlegend, opacity, base_name),
                 (sel & sched_per_trial, sched_color, True, True, max(opacity, 0.7),
                  f"{base_name} · task-switch")]
    else:
        parts = [(sel, color, False, showlegend, opacity, base_name)]
    for part_sel, part_color, is_sched, part_showlegend, part_opacity, part_name in parts:
        if part_sel.sum() == 0:
            continue
        paths = scores3[part_sel]
        gx, gy, gz = _paths_xyz(paths)
        anim_trial_groups.append((len(fig3d.data), paths))
        fig3d.add_trace(go.Scatter3d(
            x=gx, y=gy, z=gz, mode="lines", opacity=part_opacity,
            line=dict(color=part_color, width=width),
            legendgroup=label, showlegend=part_showlegend or is_sched, hoverinfo="skip",
            name=(f"{part_name} (n={int(part_sel.sum())})"
                  if (part_showlegend or is_sched) else None)))

if TRIALS_3D == "combo":
    for (stim_name, context), (label, color) in combo_style.items():
        mask = ((stim_name_per_trial == stim_name)
                & (context_per_trial == context) & trial_subset)
        _add_overlay(mask, color, label, label, showlegend=False, opacity=0.12, width=1,
                     sched_color="#8e44ad")
elif TRIALS_3D == "outcome":
    # correct / incorrect single trials, kept separate for each of the 4
    # stimulus x context classes; entries sit next to their class mean in the
    # legend and toggle one at a time. Task-switch trials of each outcome are
    # split off into their own shade of purple (deep = correct, light = incorrect).
    for (stim_name, context), (label, _) in combo_style.items():
        combo_mask = (stim_name_per_trial == stim_name) & (context_per_trial == context)
        for outcome, sel_all, ocolor, pcolor in (
                ("correct", correct_per_trial, "#2ca02c", "#6a1b9a"),
                ("incorrect", incorrect_per_trial, "#d62728", "#c77dd6")):
            sel = combo_mask & sel_all & trial_subset
            _add_overlay(sel, ocolor, label, f"{label} - {outcome}",
                         showlegend=True, opacity=0.3, width=1.5, sched_color=pcolor)

anim_mean_idx, mean_paths, mean_colors = [], [], []
for (stim_name, context), (label, color) in combo_style.items():
    mask = (stim_name_per_trial == stim_name) & (context_per_trial == context)
    if mask.sum() < 2:
        continue
    m = scores3[mask].mean(axis=0)
    anim_mean_idx.append(len(fig3d.data))
    mean_paths.append(m)
    mean_colors.append(color)
    fig3d.add_trace(go.Scatter3d(
        x=m[:, 0], y=m[:, 1], z=m[:, 2], mode="lines+markers",
        line=dict(color=color, width=6), marker=dict(size=3, color=color),
        legendgroup=label, name=f"{label} - MEAN (n={int(mask.sum())})",
        customdata=np.arange(n_bins_pca),
        hovertemplate=("time bin %{customdata}<br>PC1 %{x:.2f}<br>"
                       "PC2 %{y:.2f}<br>PC3 %{z:.2f}<extra></extra>")))

# trial-start (square) and stimulus-onset (diamond) markers, one point per
# class, each gathered into a single legend entry so it can be clicked off
_event_vis = True if EVENT_MARKERS_3D else "legendonly"
for _event, _bin, _sym, _sz in (("trial start", 0, "square", 7),
                                ("stimulus onset", onset_bin, "diamond", 8)):
    fig3d.add_trace(go.Scatter3d(
        x=[mp[_bin, 0] for mp in mean_paths], y=[mp[_bin, 1] for mp in mean_paths],
        z=[mp[_bin, 2] for mp in mean_paths], mode="markers", name=_event,
        marker=dict(size=_sz, color=mean_colors, symbol=_sym,
                    line=dict(color="black", width=1)),
        visible=_event_vis, hoverinfo="skip"))

# moving "current time" head marker, one point per class (animated below)
heads_idx = len(fig3d.data)
fig3d.add_trace(go.Scatter3d(
    x=[m[-1, 0] for m in mean_paths], y=[m[-1, 1] for m in mean_paths],
    z=[m[-1, 2] for m in mean_paths], mode="markers",
    marker=dict(size=6, color=mean_colors, symbol="circle",
                line=dict(color="black", width=1)),
    showlegend=False, hoverinfo="skip"))

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
        fdata.append(go.Scatter3d(
            x=[m[k - 1, 0] for m in mean_paths],
            y=[m[k - 1, 1] for m in mean_paths],
            z=[m[k - 1, 2] for m in mean_paths]))
        ftraces.append(heads_idx)
        frames.append(go.Frame(name=str(k), data=fdata, traces=ftraces,
                               layout=_frame_layout))
    fig3d.frames = frames

    _play = dict(frame=dict(duration=90, redraw=True),
                 transition=dict(duration=0), mode="immediate", fromcurrent=False)
    _pause = dict(frame=dict(duration=0, redraw=False),
                  transition=dict(duration=0), mode="immediate")
    _scene_menu = dict(type="buttons", direction="left", showactive=False,
                       x=0.02, y=0.02, xanchor="left", yanchor="bottom", pad=dict(r=6, t=6),
                       buttons=[dict(label="▶ Play", method="animate", args=[None, _play]),
                                dict(label="▮▮ Pause", method="animate",
                                     args=[[None], _pause])])
    _slider = dict(active=len(frames) - 1, x=0.12, y=0.02, len=0.82,
                   xanchor="left", yanchor="bottom", pad=dict(t=4, b=4),
                   currentvalue=dict(prefix="t = ", suffix=" s", xanchor="right"),
                   steps=[dict(method="animate", label=f"{bin_centers[int(f.name) - 1]:+.2f}",
                               args=[[f.name], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True),
                                                   transition=dict(duration=0))])
                          for f in frames])
    fig3d.update_layout(updatemenus=[_scene_menu], sliders=[_slider])
    _overlay_note += "; press Play or drag the slider to sweep time"

fig3d.update_layout(
    title=("PC1-PC3 trajectories, auditory cortex (AUDp/AUDv/AUDpo)<br>"
           f"<sup>mean per stimulus x context; {_overlay_note}; square = trial "
           "start, diamond = stimulus onset; time runs along each line</sup>"),
    scene=_scene,
    legend=dict(itemsizing="constant", groupclick="toggleitem"),
    width=900, height=750, margin=dict(l=0, r=0, t=70, b=0))

_out_html = Path(__file__).resolve().parent / "pca_3d_pc123.html"
fig3d.write_html(_out_html, include_plotlyjs="cdn")
print(f"wrote interactive plot -> {_out_html}")
fig3d.show()
# %%

