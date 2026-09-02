"""
PCA trajectories of single trials, split by how well the task switch was handled.
===============================================================================

THE TASK
--------
The Dynamic Routing task is a go/no-go task that alternates between two
contexts ("blocks"):

  * auditory blocks -> the mouse SHOULD lick to the sound target (sound1) and
                       ignore the visual target (vis1)
  * visual  blocks  -> the mouse should lick to vis1 instead and IGNORE that
                       very same sound1

So after every block switch the mouse has to change its answer to stimuli that
did not change at all. Handling that switch well is what we call good task
switching here.

THE TWO RECORDED POPULATIONS
----------------------------
DATA/ now holds two exports of the SAME session, one per sensory modality:

  auditory   ACx units, sound1 (target) vs sound2 (non-target)
  visual     VCx units, vis1   (target) vs vis2   (non-target)

Each one is analysed separately and the two results are shown side by side.
They are NEVER merged into a single PCA: they are different neurons, a unit
exported for the sound trials has no rows on the visual trials, so stacking all
four stimuli into one tensor would fill half of it with zeros that mean "not
exported" rather than "did not fire". PC1 would then separate sound from visual
trials for a purely clerical reason. Modality is therefore always a panel here
and never a colour. (The same argument, and a unit-count control for it, lives
in pca_both_modalities.py.)

WHAT THIS SCRIPT DOES
---------------------
  1. reads the trial tables and the spike times from the DATA folder,
  2. scores every block by how well the mouse behaved right after it switched
     into that block - once per session, from ALL its trial tables pooled, so
     both modalities are grouped by exactly the same behaviour,
  3. turns the spikes into a population activity matrix for every single trial,
     per modality, using ONLY the time points inside the window [-0.5, +1.5] s
     around stimulus onset,
  4. compresses that activity into a few principal components (PCA), one PCA
     per modality per session,
  5. draws the PCA trajectory of every single trial, with the well-handled
     blocks grouped above the badly-handled ones, so the neural difference
     between good and bad switching is easy to see.

WHAT COUNTS AS "BETTER" AND "WORSE"
-----------------------------------
The DATA folder currently holds ONE session, that is one mouse, so "better"
and "worse" are compared *within* that mouse: the blocks it switched into well
against the blocks it switched into badly.
If you later add more sessions (one sub-folder per session inside DATA, see
SECTION 2), every block of every mouse is scored in the same way and pooled,
and the panel titles then tell you which mouse each block comes from.

A WORD OF CAUTION ON THE VISUAL PANELS
--------------------------------------
The visual export has far fewer units than the auditory one (18 against 144).
Fewer neurons means noisier single trials and lower separation whatever the
brain is doing, so read auditory against visual only through the unit-matched
control in pca_both_modalities.py (F4). Within one modality, comparing its own
well-handled blocks against its own badly-handled ones is unaffected: the unit
count is identical on both sides of that comparison.

OUTPUT (written to ../results/pca_trajectories_by_performance/)
---------------------------------------------------------------
  F1_single_trial_trajectories_<modality>.png  one panel per block, every trial
  F2_population_separation.png                 what each population encodes
  F3_behavior_summary.png                      the score that defines the groups
  block_summary.csv                            one row per modality and block

HOW TO RUN IT
-------------
  python plot_pca_trajectories.py
"""
# %%
# =============================================================================
# SECTION 0 - SETTINGS (change things here, not below)
# =============================================================================

from pathlib import Path

# --- Where the data lives -----------------------------------------------------
# The folder with the CSV files exported from the Dynamic Routing dataset.
# ".." is the repository root, so this points at <repo>/DATA no matter which
# folder you start the script from.
DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"

# --- The recorded populations -------------------------------------------------
# One entry per exported modality. `target` is the stimulus the mouse should
# lick to when its own modality is the rewarded one, `other` is the non-target
# of the same modality, and `context` is the value of `rewarded_modality` in the
# blocks where this modality is the relevant one.
# A session that only has one of the two (an older, sound-only export) simply
# skips the missing one.
MODALITIES = {
    "auditory": {"label": "auditory (ACx)", "target": "sound1", "other": "sound2",
                 "context": "aud"},
    "visual":   {"label": "visual (VCx)",   "target": "vis1",   "other": "vis2",
                 "context": "vis"},
}

# --- Time window and binning --------------------------------------------------
# THE ANALYSIS WINDOW. Every spike outside this window is thrown away, so each
# trial contributes exactly the same stretch of time around stimulus onset.
TIME_WINDOW = (-0.5, 1.5)        # seconds relative to stimulus onset
BIN_WIDTH = 0.05                 # 50 ms bins -> 40 time points per trial
SMOOTHING_SIGMA_BINS = 1.5       # Gaussian smoothing of the firing rates, in bins

# --- PCA ----------------------------------------------------------------------
N_PCS = 3                        # how many principal components to keep
PLOT_PCS = (0, 1)                # which two PCs to draw (0 = PC1, 1 = PC2)

# --- Behaviour ----------------------------------------------------------------
# How much of a block counts as "just after the switch". The number counts ALL
# trials of the session; pooling every trial table of a session (all four
# stimuli, not just the sounds) means nearly all of them are available, so the
# score is now built on about twice as many trials as it used to be.
N_TRIALS_AFTER_SWITCH = 30
EXCLUDE_INSTRUCTION_TRIALS = True   # free rewarded trials given after a switch
EXCLUDE_OPTO_TRIALS = True          # laser trials perturb the activity itself

# The two groups are made by cutting the ranked blocks in half, so they always
# exist - even when every block was handled equally well and the cut falls
# between two scores that differ by a single trial. A well-trained mouse in a
# short session is exactly that case. When the gap at the cut is smaller than
# this, the script says so instead of presenting a contrast it does not have.
MIN_MEANINGFUL_SPLIT_GAP = 0.05

# The first block of a session follows no switch, so strictly speaking it says
# nothing about switching. It does show how well the mouse was following the
# rule at the start though, and that is often its weakest moment, so it is kept
# by default and marked "session start" in the figures.
# Set this to False to look at genuine switches only.
SCORE_FIRST_BLOCK_TOO = True

# --- Plotting -----------------------------------------------------------------
MAX_PANELS_PER_GROUP = 6         # at most this many blocks drawn per group
MAX_TRIALS_TO_DRAW = None        # None = draw every trial; set e.g. 30 to thin out

# --- Colours (from a validated categorical palette) ---------------------------
# Colour means STIMULUS ROLE or BEHAVIOURAL GROUP everywhere in this file.
# Which population was recorded is carried by the panels, never by a hue.
COLOR_TARGET = "#2a78d6"         # blue   - the target stimulus (sound1 / vis1)
COLOR_OTHER = "#eb6834"          # orange - the non-target      (sound2 / vis2)
COLOR_BETTER = "#2a78d6"         # blue   - blocks switched into well
COLOR_WORSE = "#eb6834"          # orange - blocks switched into badly
COLOR_REFERENCE = "#8a8985"      # grey   - reference / all-blocks curves
COLOR_GRID = "#d8d7d2"           # recessive grid / axes
COLOR_TEXT_SECONDARY = "#52514e"

# --- Where the figures go -----------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "pca_trajectories_by_performance"


# %%
# =============================================================================
# SECTION 1 - IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # save figures to file, no screen needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   # only used to build the legends
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


# %%
# =============================================================================
# SECTION 2 - READING THE CSV FILES
# =============================================================================
#
# The DATA folder holds pairs of files, one pair per stimulus:
#
#   sound1_trials.csv       one row per trial  (id, stim_start_time, block_index,
#                                               rewarded_modality, is_correct, ...)
#   sound1_spike_times.csv  one row per spike  (unit_id, trial_id, spike_time)
#
# `spike_time` is already measured relative to the stimulus onset of its trial,
# which is exactly what this analysis needs.
#
# The spike files of one modality only contain that modality's units, so they
# are read by name (sound1/sound2, vis1/vis2) and never by a glob - a glob would
# mix ACx and VCx units into one tensor, which is the one thing this analysis
# must not do.
#
# If you get more sessions later, put each one in its own sub-folder of DATA
# (DATA/713655_2024-08-09/sound1_trials.csv, ...). The code below finds them
# automatically and treats every sub-folder as a separate session.

def find_sessions(data_dir):
    """
    Return a list of (session_name, folder) pairs.

    A folder counts as a session when it directly contains "*_trials.csv" files.
    """
    sub_folders = sorted(folder for folder in data_dir.iterdir()
                         if folder.is_dir() and any(folder.glob("*_trials.csv")))
    if sub_folders:
        return [(folder.name, folder) for folder in sub_folders]

    if any(data_dir.glob("*_trials.csv")):
        return [(data_dir.name, data_dir)]   # a single session, files sit in DATA

    return []


def load_modality_tables(folder, spec):
    """
    Read the two trials/spikes CSV pairs of ONE modality and stack them.

    Returns (trials, spikes) with `trials` sorted by trial id and one row per
    spike in `spikes`, or (None, None) when this session does not have that
    modality exported.
    """
    trials_parts, spikes_parts = [], []

    for stimulus_name in (spec["target"], spec["other"]):
        trials_file = folder / f"{stimulus_name}_trials.csv"
        spikes_file = folder / f"{stimulus_name}_spike_times.csv"
        if not (trials_file.exists() and spikes_file.exists()):
            return None, None
        trials_parts.append(pd.read_csv(trials_file))
        spikes_parts.append(pd.read_csv(spikes_file))

    trials = (pd.concat(trials_parts, ignore_index=True)
                .sort_values("id").reset_index(drop=True))
    spikes = pd.concat(spikes_parts, ignore_index=True)
    return trials, spikes


def load_session_behaviour(folder):
    """
    Every trial of one session, pooled from ALL of its trial tables.

    Behaviour needs no spikes, so here the glob IS the right thing: the more
    stimuli go in, the better the estimate of what the mouse did just after a
    switch. Trials shared by two tables cannot happen (a trial has one
    stim_name), but ids are de-duplicated anyway so a folder holding an extra
    export of the same stimulus cannot count a trial twice.
    """
    parts = [pd.read_csv(f) for f in sorted(folder.glob("*_trials.csv"))]
    if not parts:
        return None
    return (pd.concat(parts, ignore_index=True)
              .drop_duplicates(subset="id")
              .sort_values("id").reset_index(drop=True))


def boolean_column(trials, column_name):
    """Return a trials column as a plain True/False array (missing -> False)."""
    if column_name not in trials.columns:
        return np.zeros(len(trials), dtype=bool)
    return trials[column_name].fillna(False).to_numpy().astype(bool)


def build_trial_tensor(trials, spikes, bin_edges):
    """
    Count the spikes of every unit, in every time bin, on every trial.

    Spikes falling outside [-0.5, 1.5] s (or belonging to a trial we dropped)
    are simply not counted - that is the requested window restriction.

    Returns (tensor, unit_ids), where tensor has the shape
    (n_trials, n_bins, n_units) and holds firing rates in spikes per second.
    """
    trial_ids = trials["id"].to_numpy()
    unit_ids = np.sort(spikes["unit_id"].unique())

    # where each trial / unit sits in the tensor
    trial_position = {trial_id: i for i, trial_id in enumerate(trial_ids)}
    unit_position = {unit_id: i for i, unit_id in enumerate(unit_ids)}

    n_bins = len(bin_edges) - 1
    counts = np.zeros((len(trial_ids), n_bins, len(unit_ids)), dtype=np.float32)

    # keep the spikes that belong to a kept trial AND lie inside the window
    inside_window = (
        spikes["spike_time"].between(bin_edges[0], bin_edges[-1], inclusive="left")
        & spikes["trial_id"].isin(trial_position.keys())
    )
    kept = spikes[inside_window]

    # which trial / time bin / unit does each spike belong to?
    trial_index = kept["trial_id"].map(trial_position).to_numpy()
    unit_index = kept["unit_id"].map(unit_position).to_numpy()
    bin_index = np.floor((kept["spike_time"].to_numpy() - bin_edges[0]) / BIN_WIDTH).astype(int)

    # add 1 to the right cell for every spike (np.add.at also handles repeats)
    np.add.at(counts, (trial_index, bin_index, unit_index), 1)

    return counts / BIN_WIDTH, unit_ids   # counts -> spikes per second


# %%
# =============================================================================
# SECTION 3 - BEHAVIOUR: how well was each block switch handled?
# =============================================================================

def score_blocks(trials, n_trials_after_switch=N_TRIALS_AFTER_SWITCH):
    """
    Give every block a score between 0 and 1: the fraction of correct answers
    the mouse gave in the trials right after it entered that block.

    "Correct" is already worked out by the dataset (the `is_correct` column)
    and it knows the rule of the current block:

        auditory block: lick to sound1 = correct, lick to vis1 = wrong
        visual block:   lick to vis1   = correct, lick to sound1 = wrong
                        (the same sound1 that had to be licked to before)

    A high score therefore means "the mouse adapted quickly to the new rule".
    This is computed once per session, on the pooled trial table, so both
    modalities end up with exactly the same grouping of blocks.

    The first block of a session is scored as well, but it follows no switch,
    so it is flagged with follows_a_switch = False and left out later on.

    Returns a DataFrame with one row per block.
    """
    countable = np.ones(len(trials), dtype=bool)
    if EXCLUDE_INSTRUCTION_TRIALS:
        countable &= ~boolean_column(trials, "is_instruction")
    if EXCLUDE_OPTO_TRIALS:
        countable &= ~boolean_column(trials, "is_opto")
    countable &= ~boolean_column(trials, "is_repeat")

    just_after_switch = trials["trial_index_in_block"].to_numpy() < n_trials_after_switch
    is_correct = boolean_column(trials, "is_correct")
    is_response = boolean_column(trials, "is_response")
    is_sound1 = (trials["stim_name"] == "sound1").to_numpy()
    block_index = trials["block_index"].to_numpy()

    rows = []
    for position, block in enumerate(np.unique(block_index)):
        in_block = block_index == block
        scored = in_block & countable & just_after_switch
        if scored.sum() == 0:
            continue

        # response rate to sound1: high in auditory blocks and low in visual
        # blocks if the mouse is following the rule of the new block
        sound1_after_switch = scored & is_sound1

        rows.append({
            "block_index": block,
            "rewarded_modality": trials.loc[in_block, "rewarded_modality"].iloc[0],
            "follows_a_switch": position > 0,
            "switch_score": float(is_correct[scored].mean()),
            "n_trials_scored": int(scored.sum()),
            "sound1_response_rate_after_switch": (
                float(is_response[sound1_after_switch].mean())
                if sound1_after_switch.sum() else np.nan
            ),
            "whole_block_score": float(is_correct[in_block & countable].mean()),
        })

    return pd.DataFrame(rows)


# %%
# =============================================================================
# SECTION 4 - POPULATION ACTIVITY -> PCA TRAJECTORIES
# =============================================================================

def smooth_and_standardise(tensor, sigma_bins=SMOOTHING_SIGMA_BINS):
    """
    Make the units comparable to each other before running PCA.

    1. smooth along time, so single-trial rates are not raw 0/20/40 Hz steps,
    2. z-score each unit (subtract its mean, divide by its standard deviation),
       otherwise one high-firing neuron would dominate every component,
    3. drop units that never fire (their standard deviation is 0).
    """
    tensor = gaussian_filter1d(tensor, sigma=sigma_bins, axis=1, mode="nearest")

    unit_means = tensor.mean(axis=(0, 1))
    unit_stds = tensor.std(axis=(0, 1))

    active_units = unit_stds > 0
    tensor = tensor[:, :, active_units]
    return (tensor - unit_means[active_units]) / unit_stds[active_units]


def run_pca_on_trials(tensor, n_components=N_PCS):
    """
    Compress the population activity down to a few principal components.

    The input has the shape (n_trials, n_bins, n_units). PCA needs a flat
    table, so all trials and time points are stacked into rows, the components
    are fitted on that table, and the result is folded back into single trials.

    PCA is fitted once per session AND per modality: different sessions record
    different neurons, and the two modality exports are different neurons too,
    so neither shares a space with the other.

    Returns (scores, explained_variance_ratio) with scores of shape
    (n_trials, n_bins, n_components).
    """
    n_trials, n_bins, n_units = tensor.shape
    flat_activity = tensor.reshape(n_trials * n_bins, n_units)

    pca = PCA(n_components=min(n_components, n_units))
    flat_scores = pca.fit_transform(flat_activity)

    return flat_scores.reshape(n_trials, n_bins, -1), pca.explained_variance_ratio_


def separation_over_time(scores, group_a, group_b):
    """
    How different are two sets of trials, at each moment of the trial?

    We take the distance between the two average trajectories and divide it by
    the trial-to-trial spread. Dividing makes the number comparable between
    blocks and between mice: it says "how many trial-to-trial standard
    deviations apart are these two conditions".

    `group_a` and `group_b` are True/False masks over the trials.
    """
    if group_a.sum() < 2 or group_b.sum() < 2:
        return np.full(scores.shape[1], np.nan)

    a, b = scores[group_a], scores[group_b]
    distance = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0), axis=1)
    spread = np.sqrt(0.5 * (a.var(axis=0).sum(axis=1) + b.var(axis=0).sum(axis=1)))
    return distance / np.where(spread > 0, spread, np.nan)


def average_curve(curves):
    """Average a list of time courses, ignoring the ones that are all NaN."""
    usable = [curve for curve in curves if not np.all(np.isnan(curve))]
    if not usable:
        return None, None, 0
    stacked = np.vstack(usable)
    mean_curve = np.nanmean(stacked, axis=0)
    sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(len(stacked))
    return mean_curve, sem_curve, len(stacked)


def context_separation(run, block_indices=None):
    """
    How differently is the SAME target stimulus represented in the two contexts?

    The stimulus does not change, only its meaning does, so this is the neural
    trace of the context the mouse thinks it is in. Restrict it to a set of
    blocks to ask the question of one behavioural group only; both contexts have
    to be present among those blocks, otherwise there is nothing to compare and
    the curve comes back as NaN.
    """
    selected = (np.ones(len(run["is_target"]), dtype=bool) if block_indices is None
                else np.isin(run["trial_block_index"], list(block_indices)))
    return separation_over_time(
        run["scores"],
        selected & run["is_target"] & run["is_own_context"],
        selected & run["is_target"] & ~run["is_own_context"])


# %%
# =============================================================================
# SECTION 5 - PUT IT TOGETHER: one session -> one run per modality
# =============================================================================

def process_session(session_name, folder, bin_edges, bin_centers):
    """
    Load one session and run a separate PCA on each modality it holds.

    Returns (block_scores, runs): the behaviour of the session, scored once on
    all its trials, and one "run" per modality carrying that modality's PCA
    scores plus the masks needed to slice them by stimulus, context and block.
    """
    behaviour = load_session_behaviour(folder)
    if behaviour is None:
        print(f"  [skip] {session_name}: no *_trials.csv found")
        return None, []

    # ---- behaviour, once per session, before any trial is dropped -----------
    block_scores = score_blocks(behaviour)
    trials_per_block = block_scores["n_trials_scored"].mean()
    print(f"  {session_name}: behaviour from {len(behaviour)} trials, "
          f"{trials_per_block:.0f} of the first {N_TRIALS_AFTER_SWITCH} trials "
          f"scored per block")

    # ---- one PCA per modality, on that modality's units only ----------------
    runs = []
    for name, spec in MODALITIES.items():
        trials, spikes = load_modality_tables(folder, spec)
        if trials is None:
            print(f"    [skip] {name}: no {spec['target']}/{spec['other']} "
                  "CSV pair in this session")
            continue

        # keep the trials we can align and want to plot
        # .copy() because a numpy array handed over by pandas can be read-only
        keep = trials["stim_start_time"].notna().to_numpy().copy()
        if EXCLUDE_OPTO_TRIALS:
            keep &= ~boolean_column(trials, "is_opto")
        trials = trials[keep].reset_index(drop=True)

        # spikes -> firing rates -> PCA
        tensor, unit_ids = build_trial_tensor(trials, spikes, bin_edges)
        tensor = smooth_and_standardise(tensor)
        scores, explained_variance = run_pca_on_trials(tensor)
        print(f"    {name:<9} {len(trials):3d} trials, {tensor.shape[2]:3d} units, "
              f"PC1-PC{scores.shape[2]} explain "
              f"{100 * explained_variance.sum():.0f}% of the variance")

        runs.append({
            "session_name": session_name,
            "modality": name,
            "modality_label": spec["label"],
            "spec": spec,
            "n_units": int(tensor.shape[2]),
            "bin_centers": bin_centers,
            "explained_variance": explained_variance,
            "scores": scores,
            "is_target": (trials["stim_name"] == spec["target"]).to_numpy(),
            "is_own_context": (trials["rewarded_modality"] == spec["context"]).to_numpy(),
            "trial_block_index": trials["block_index"].to_numpy(),
        })

    return block_scores, runs


def blocks_of_run(run, block_scores):
    """
    Cut one run into blocks: the trajectories of each block's own trials,
    plus the behaviour of that block.

    Every block dictionary already carries everything the plotting functions
    need, so they never have to touch the raw data again.
    """
    blocks = []
    for _, block_row in block_scores.iterrows():
        in_block = run["trial_block_index"] == block_row["block_index"]
        if in_block.sum() == 0:
            continue

        blocks.append({
            "session_name": run["session_name"],
            "modality": run["modality"],
            "modality_label": run["modality_label"],
            "spec": run["spec"],
            "block_index": int(block_row["block_index"]),
            "rewarded_modality": block_row["rewarded_modality"],
            "follows_a_switch": bool(block_row["follows_a_switch"]),
            "switch_score": float(block_row["switch_score"]),
            "whole_block_score": float(block_row["whole_block_score"]),
            "sound1_response_rate_after_switch": float(block_row["sound1_response_rate_after_switch"]),
            "n_units": run["n_units"],
            "n_trials": int(in_block.sum()),
            "bin_centers": run["bin_centers"],
            "explained_variance": run["explained_variance"],
            # the trajectories of this block's trials, and which stimulus each was
            "scores": run["scores"][in_block],
            "is_target": run["is_target"][in_block],
            # how well this block's population tells target from non-target
            "stimulus_separation": separation_over_time(
                run["scores"][in_block], run["is_target"][in_block],
                ~run["is_target"][in_block]),
        })

    return blocks


# %%
# =============================================================================
# SECTION 6 - FIGURES
# =============================================================================

def style_axis(ax):
    """A quiet, uncluttered axis: no top/right frame, faint grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def block_label(block):
    """The short readable name built in main(), e.g. 'block 3 (vis-rewarded)'."""
    return block["label"]


def plot_single_trial_trajectories(better, worse, run, output_path):
    """
    FIGURE 1 - one panel per block, one thin line per trial, for one modality.

    Every thin line is the trajectory of a single trial through the PCA space,
    from -0.5 s to +1.5 s around stimulus onset. Colour tells which stimulus was
    presented. The thick lines are the averages over all trials of that
    stimulus, and the dot marks stimulus onset (t = 0).

    The top rows are the blocks the mouse switched into well, the bottom rows
    the blocks it switched into badly.
    """
    x_pc, y_pc = PLOT_PCS
    spec = run["spec"]
    groups = [(better[:MAX_PANELS_PER_GROUP], "SWITCHED IN WELL"),
              (worse[:MAX_PANELS_PER_GROUP], "SWITCHED IN BADLY")]

    n_cols = min(3, max(len(groups[0][0]), len(groups[1][0]), 1))
    rows_per_group = [int(np.ceil(len(group) / n_cols)) for group, _ in groups]
    n_rows = max(sum(rows_per_group), 1)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(4.3 * n_cols, 4.1 * n_rows))
    for ax in axes.ravel():
        ax.axis("off")   # panels we do not fill stay empty

    first_row_of_group = [0, rows_per_group[0]]
    for group_number, (group, group_title) in enumerate(groups):
        for position, block in enumerate(group):
            ax = axes[first_row_of_group[group_number] + position // n_cols,
                      position % n_cols]
            ax.axis("on")
            style_axis(ax)

            scores = block["scores"]
            is_target = block["is_target"]

            # -- every single trial, drawn faintly --------------------------
            trials_to_draw = np.arange(len(scores))
            if MAX_TRIALS_TO_DRAW is not None and len(trials_to_draw) > MAX_TRIALS_TO_DRAW:
                trials_to_draw = np.linspace(0, len(scores) - 1, MAX_TRIALS_TO_DRAW).astype(int)
            # the more trials there are, the fainter each one has to be
            trial_alpha = float(np.clip(15 / max(len(trials_to_draw), 1), 0.08, 0.35))

            for trial in trials_to_draw:
                ax.plot(scores[trial, :, x_pc], scores[trial, :, y_pc],
                        color=COLOR_TARGET if is_target[trial] else COLOR_OTHER,
                        linewidth=0.6, alpha=trial_alpha, zorder=1)

            # -- average trajectory per stimulus, dot at stimulus onset -----
            onset_bin = int(np.argmin(np.abs(block["bin_centers"])))
            for selection, color, style in ((is_target, COLOR_TARGET, "-"),
                                            (~is_target, COLOR_OTHER, "--")):
                if selection.sum() == 0:
                    continue
                mean_trajectory = scores[selection].mean(axis=0)
                ax.plot(mean_trajectory[:, x_pc], mean_trajectory[:, y_pc],
                        color=color, linewidth=2, linestyle=style, zorder=3)
                ax.plot(mean_trajectory[onset_bin, x_pc], mean_trajectory[onset_bin, y_pc],
                        "o", color=color, markersize=8, markeredgecolor="white",
                        markeredgewidth=2, zorder=4)

            ax.set_title(f"{block_label(block)}\n"
                         f"switch score = {block['switch_score']:.2f}   |   "
                         f"{block['n_trials']} trials",
                         fontsize=9)
            ax.set_xlabel(f"PC{x_pc + 1}", fontsize=9)
            ax.set_ylabel(f"PC{y_pc + 1}", fontsize=9)

            # the group name, written once next to the first panel of the group
            if position == 0:
                ax.text(-0.28, 0.5, group_title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold",
                        color=COLOR_TEXT_SECONDARY)

    fig.legend(handles=[
        Line2D([], [], color=COLOR_TARGET, linewidth=2,
               label=f"{spec['target']} (target)"),
        Line2D([], [], color=COLOR_OTHER, linewidth=2, linestyle="--",
               label=f"{spec['other']} (non-target)"),
        Line2D([], [], color=COLOR_TEXT_SECONDARY, marker="o", linestyle="",
               markersize=8, label="stimulus onset (t = 0)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"Single-trial population trajectories - {run['modality_label']}, "
        f"{run['n_units']} units\n"
        f"{TIME_WINDOW[0]} to +{TIME_WINDOW[1]} s around stimulus onset; thin line = one "
        "trial, thick line = average per stimulus,\nblocks sorted by how well the mouse "
        "handled the switch",
        fontsize=12)
    fig.tight_layout(rect=[0.02, 0.035, 1, 0.92])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {output_path}")


def plot_population_separation(runs, blocks_by_run, output_path):
    """
    FIGURE 2 - the quantitative comparison between the two groups of blocks,
    one row per recorded population.

    Left  : how well the population tells the target from the non-target inside
            each block, well-handled blocks against badly-handled ones.
    Right : how differently the SAME target stimulus is represented in the two
            contexts - the neural trace of the context - computed over all
            blocks and then separately within each behavioural group. Both
            contexts have to be present in a group for its curve to exist, so a
            group made only of auditory (or only of visual) blocks is left out.
    """
    fig, axes = plt.subplots(len(runs), 2, figsize=(12, 4.4 * len(runs)),
                             squeeze=False)

    for row, run in enumerate(runs):
        spec = run["spec"]
        time = run["bin_centers"]
        blocks = blocks_by_run[row]
        better = [b for b in blocks if b["is_better"]]
        worse = [b for b in blocks if not b["is_better"]]

        # -- left panel: stimulus separation, one thin line per block --------
        ax = axes[row][0]
        style_axis(ax)
        for group, color, style, label in ((better, COLOR_BETTER, "-", "switched in well"),
                                           (worse, COLOR_WORSE, "--", "switched in badly")):
            for block in group:
                ax.plot(time, block["stimulus_separation"], color=color, linestyle=style,
                        linewidth=1, alpha=0.5)
            mean_curve, sem_curve, n_used = average_curve([b["stimulus_separation"] for b in group])
            if mean_curve is None:
                continue
            ax.plot(time, mean_curve, color=color, linestyle=style, linewidth=2.5,
                    label=f"{label} (n = {n_used} blocks)")
            ax.fill_between(time, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=color, alpha=0.15, linewidth=0)
        ax.axvline(0, color=COLOR_GRID, linewidth=1)
        # a block whose own modality is the rewarded one separates its two
        # stimuli much more strongly (the mouse only licks there), so blocks are
        # best compared with blocks of the same context - the panel titles of F1
        # name them.
        ax.set_title(f"{run['modality_label']}, {run['n_units']} units\n"
                     f"{spec['target']} versus {spec['other']}, within each block",
                     fontsize=10)
        ax.set_xlabel("time from stimulus onset (s)", fontsize=9)
        ax.set_ylabel("separation\n(population distance / trial spread)", fontsize=9)
        ax.legend(frameon=False, fontsize=9)

        # -- right panel: context separation of the target stimulus ----------
        ax = axes[row][1]
        style_axis(ax)
        curves = [
            (context_separation(run), COLOR_REFERENCE, "-", "all blocks"),
            (context_separation(run, [b["block_index"] for b in better]),
             COLOR_BETTER, "-", "blocks switched into well"),
            (context_separation(run, [b["block_index"] for b in worse]),
             COLOR_WORSE, "--", "blocks switched into badly"),
        ]
        for curve, color, style, label in curves:
            if np.all(np.isnan(curve)):
                continue    # that group has blocks of one context only
            ax.plot(time, curve, color=color, linestyle=style, linewidth=2, label=label)
        ax.axvline(0, color=COLOR_GRID, linewidth=1)
        ax.set_title(f"{run['modality_label']}\n"
                     f"{spec['target']} in auditory versus visual blocks",
                     fontsize=10)
        ax.set_xlabel("time from stimulus onset (s)", fontsize=9)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("What each population encodes: stimulus (left) and context (right)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {output_path}")


def plot_behavior_summary(behaviour_blocks, split_score, split_gap, output_path):
    """
    FIGURE 3 - the behaviour that defines the two groups.

    One bar per block: the fraction of correct answers in the trials right
    after the switch into that block. The dashed line is where the blocks were
    cut into "switched in well" and "switched in badly". There is one bar per
    block, not per modality: the score comes from the mouse, and both recorded
    populations are grouped by it.
    """
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(behaviour_blocks)), 4.6))
    style_axis(ax)

    positions = np.arange(len(behaviour_blocks))
    values = [block["switch_score"] for block in behaviour_blocks]
    colors = [COLOR_BETTER if block["is_better"] else COLOR_WORSE
              for block in behaviour_blocks]

    ax.bar(positions, values, color=colors, width=0.7)
    ax.axhline(split_score, color=COLOR_TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.text(len(behaviour_blocks) - 0.4, split_score, f"  split at {split_score:.2f}",
            va="center", fontsize=8, color=COLOR_TEXT_SECONDARY)

    # the bars are few, so a direct label on each one is fine here
    for position, value in zip(positions, values):
        ax.text(position, value, f"{value:.2f}", ha="center", va="bottom",
                fontsize=8, color=COLOR_TEXT_SECONDARY)

    ax.set_xticks(positions)
    ax.set_xticklabels([block_label(block).replace("  ", "\n")
                        for block in behaviour_blocks],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"fraction correct in the first\n{N_TRIALS_AFTER_SWITCH} trials of the block",
                  fontsize=9)
    ax.legend(handles=[
        Line2D([], [], color=COLOR_BETTER, linewidth=8, label="switched in well"),
        Line2D([], [], color=COLOR_WORSE, linewidth=8, label="switched in badly"),
    ], frameon=False, fontsize=9, loc="lower left")

    # The bars are the whole argument of this script, so when they do not
    # actually differ the figure has to say it, right where they are read.
    warn = split_gap < MIN_MEANINGFUL_SPLIT_GAP
    ax.set_title("How well each block switch was handled", fontsize=12,
                 pad=20 if warn else None)
    if warn:
        ax.text(0.5, 1.01,
                f"the two groups differ by only {split_gap:.3f} - this split is a "
                "ranking artefact, not a behavioural difference",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                color=COLOR_WORSE)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {output_path}")


# %%
# =============================================================================
# SECTION 7 - MAIN PROGRAM
# =============================================================================

def label_for(block, n_sessions):
    """A readable name for a block; the session is only worth naming when
    there is more than one of them."""
    context = "aud-rewarded" if block["rewarded_modality"] == "aud" else "vis-rewarded"
    if not block["follows_a_switch"]:
        context += ", session start"
    prefix = f"{block['session_name']}  " if n_sessions > 1 else ""
    return f"{prefix}block {block['block_index']} ({context})"


def main():
    # The bin edges define the analysis window: everything outside [-0.5, 1.5]
    # is excluded from every trial. linspace (rather than arange) guarantees
    # that the last edge lands exactly on the end of the window.
    n_bins = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_WIDTH))
    bin_edges = np.linspace(TIME_WINDOW[0], TIME_WINDOW[1], n_bins + 1)
    bin_centers = bin_edges[:-1] + BIN_WIDTH / 2

    if not DATA_DIR.is_dir():
        raise SystemExit(f"Data folder not found: {DATA_DIR}\n"
                         "Edit DATA_DIR at the top of this script.")

    sessions = find_sessions(DATA_DIR)
    if not sessions:
        raise SystemExit(f"No '*_trials.csv' files found in {DATA_DIR} "
                         "or in its sub-folders.")

    # ---- 1. read every session: behaviour once, PCA once per modality -------
    print(f"Reading {len(sessions)} session(s) from {DATA_DIR} ...")
    runs, behaviour_blocks, behaviour_by_session = [], [], {}
    for session_name, folder in sessions:
        block_scores, session_runs = process_session(session_name, folder,
                                                     bin_edges, bin_centers)
        if not session_runs:
            continue
        runs.extend(session_runs)
        behaviour_by_session[session_name] = block_scores
        for _, row in block_scores.iterrows():
            behaviour_blocks.append({
                "session_name": session_name,
                "block_index": int(row["block_index"]),
                "rewarded_modality": row["rewarded_modality"],
                "follows_a_switch": bool(row["follows_a_switch"]),
                "switch_score": float(row["switch_score"]),
            })
    if not runs:
        raise SystemExit("No modality could be loaded from any session.")

    # Strictly speaking only blocks that follow a switch say something about
    # task switching - see SCORE_FIRST_BLOCK_TOO in the settings.
    n_first_blocks = sum(1 for b in behaviour_blocks if not b["follows_a_switch"])
    if not SCORE_FIRST_BLOCK_TOO:
        behaviour_blocks = [b for b in behaviour_blocks if b["follows_a_switch"]]
        print(f"  ({n_first_blocks} first block(s) left out: no switch before them)")
    elif n_first_blocks:
        print(f"  ({n_first_blocks} first block(s) kept and marked 'session start')")
    if len(behaviour_blocks) < 2:
        raise SystemExit("Need at least 2 blocks that follow a switch to compare.")

    # ---- 2. split the blocks into well and badly handled switches -----------
    # The split is made on the behaviour alone, so both modalities are grouped
    # in exactly the same way and can be read against each other.
    # The blocks are sorted by their score and cut in half. Cutting by rank
    # (rather than by the median value) keeps the two groups the same size even
    # when several blocks happen to share the same score.
    behaviour_blocks.sort(key=lambda block: -block["switch_score"])   # best first
    half = int(np.ceil(len(behaviour_blocks) / 2))
    for rank, block in enumerate(behaviour_blocks):
        block["is_better"] = rank < half
        block["label"] = label_for(block, len(sessions))
    # where the cut falls, half way between the two groups (only used for F3)
    split_gap = (behaviour_blocks[half - 1]["switch_score"]
                 - behaviour_blocks[half]["switch_score"])
    split_score = behaviour_blocks[half]["switch_score"] + split_gap / 2
    grouping = {(b["session_name"], b["block_index"]): b for b in behaviour_blocks}

    print("\nHow well each switch was handled (1.00 = every answer correct):")
    for block in behaviour_blocks:
        group = "well " if block["is_better"] else "badly"
        print(f"  {block_label(block):<40} switch score = {block['switch_score']:.3f}  [{group}]")

    # A split is only worth interpreting when the two groups really differ. If
    # they do not, the figures still get drawn - they are useful in themselves -
    # but nothing about "good versus bad switching" may be read into them.
    if split_gap < MIN_MEANINGFUL_SPLIT_GAP:
        print(f"\n  [warning] the two groups are separated by only {split_gap:.3f} "
              f"in switch score.\n"
              f"            The mouse handled these blocks about equally well, so the "
              f"well/badly\n"
              f"            split here is a ranking artefact rather than a behavioural "
              f"difference.\n"
              f"            Compare the groups only when this gap is clearly above "
              f"{MIN_MEANINGFUL_SPLIT_GAP}.")

    # ---- 3. cut every run into blocks and carry the grouping over -----------
    blocks_by_run = []
    for run in runs:
        # the same table the behaviour came from, minus any block the split
        # above dropped (the first block of a session, when it is not scored)
        session_scores = behaviour_by_session[run["session_name"]]
        kept = [(run["session_name"], int(b)) in grouping
                for b in session_scores["block_index"]]
        blocks = blocks_of_run(run, session_scores[kept])
        for block in blocks:
            reference = grouping[(block["session_name"], block["block_index"])]
            block["is_better"] = reference["is_better"]
            block["label"] = reference["label"]
        blocks.sort(key=lambda block: -block["switch_score"])
        blocks_by_run.append(blocks)

    # ---- 4. draw the figures ------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nDrawing figures ...")
    for run, blocks in zip(runs, blocks_by_run):
        better = [b for b in blocks if b["is_better"]]
        worse = [b for b in blocks if not b["is_better"]]
        plot_single_trial_trajectories(
            better, worse, run,
            OUTPUT_DIR / f"F1_single_trial_trajectories_{run['modality']}.png")
    plot_population_separation(runs, blocks_by_run,
                               OUTPUT_DIR / "F2_population_separation.png")
    plot_behavior_summary(behaviour_blocks, split_score, split_gap,
                          OUTPUT_DIR / "F3_behavior_summary.png")

    # ---- 5. save the numbers behind the figures -----------------------------
    after_onset = bin_centers > 0
    summary = pd.DataFrame([{
        "session_name": block["session_name"],
        "modality": block["modality"],
        "region": block["modality_label"],
        "block_index": block["block_index"],
        "rewarded_modality": block["rewarded_modality"],
        "group": "switched in well" if block["is_better"] else "switched in badly",
        "switch_score": block["switch_score"],
        "whole_block_score": block["whole_block_score"],
        "sound1_response_rate_after_switch": block["sound1_response_rate_after_switch"],
        "n_trials": block["n_trials"],
        "n_units": block["n_units"],
        "variance_explained_by_kept_pcs": float(np.sum(block["explained_variance"])),
        "mean_stimulus_separation_after_onset": float(
            np.nanmean(block["stimulus_separation"][after_onset])),
    } for blocks in blocks_by_run for block in blocks])
    summary.to_csv(OUTPUT_DIR / "block_summary.csv", index=False)
    print(f"  saved {OUTPUT_DIR / 'block_summary.csv'}")
    print("\nDone.")


# %%
if __name__ == "__main__":
    main()

# %%
