"""
neural_subspace_projection.py

Builds a low-dimensional "neural state space" from the synthetic
Dynamic-Routing ensemble spiking data generated in
`dynamic_routing_hmm_sweep__3_.py`, then projects INDIVIDUAL SINGLE TRIALS
into that shared subspace so trajectories can be compared across trial
types -- in the spirit of Mante, Sussillo, Shenoy & Newsome (2013).

Two comparisons are produced:

  1. Correct rejects vs. correct licks (hits) vs. false alarms (vs. misses)
     -- do these trial-outcome types follow distinct paths through the
     subspace?

  2. Correct "aud1" (aud_target) responses in the auditory (context-matched)
     block vs. incorrect aud_target responses -- i.e. false alarms -- to the
     identical physical stimulus when it appears in the visual
     (context-mismatched) block. Same stimulus, different context/outcome:
     does the trajectory differ?

Method (state-space construction)
----------------------------------
Following Mante et al. 2013's general logic: single-trial spike counts are
noisy, so the *subspace itself* is estimated from trial-averaged
("condition-averaged") firing-rate trajectories, which denoise the shared
task-related structure. Individual single trials are then projected into
that fixed, pre-estimated subspace, so both single-trial variability and
condition-average trajectories can be visualized in the same coordinates.

We build the subspace with PCA on z-scored condition-averaged PSTHs (the
classic "denoised PCA" state-space approach). We additionally compute a
simple regression-style "choice axis" (mean-difference between licked vs.
not-licked activity) as a lightweight analogue of Mante's targeted-
dimensionality-reduction regression axes, since it is often more
interpretable for a specific binary (correct vs. incorrect) comparison than
unsupervised PCA alone.

Only numpy / scipy / matplotlib are used, consistent with the parent script.
"""

import os
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; set before any pyplot import
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d


# ==========================================================================
# 0. Import the synthetic-data generator from the provided file, by path
#    (its filename has spaces/parens so it can't be `import`-ed normally).
# ==========================================================================

_SOURCE_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "dynamic_routing_hmm_sweep.py"),
    "/mnt/user-data/uploads/dynamic_routing_hmm_sweep.py",
    "dynamic_routing_hmm_sweep.py",
]

_dr_module = None  # lazily loaded -- see _get_source_module()


def _load_source_module():
    for path in _SOURCE_CANDIDATES:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("dr_hmm_sweep", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)   # runs module-level defs only
            return mod
    raise FileNotFoundError(
        "Could not find dynamic_routing_hmm_sweep.py. Place it next to "
        "this script, or edit _SOURCE_CANDIDATES with its path."
    )


def _get_source_module():
    """Loaded on first use (by simulate_session), NOT at import time --
    this file only actually needs the synthetic-data generator for its own
    demo/simulation; other scripts (e.g. neural_subspace_real_data.py) that
    import this module just to reuse the subspace/plotting code shouldn't
    be forced to have dynamic_routing_hmm_sweep__3_.py available at all."""
    global _dr_module
    if _dr_module is None:
        _dr_module = _load_source_module()
    return _dr_module


# ==========================================================================
# 1. Simulate a session and label every trial by stimulus / context / outcome
# ==========================================================================

def simulate_session(scenario="clear", num_blocks=16, trials_per_block=50,
                      num_neurons=16, bin_size_s=0.02, seed=0):
    """Simulate one Dynamic-Routing session using the parent script's
    generators. `scenario` in {'clear', 'ambiguous'} selects how separable
    the ground-truth decision state is (see SCENARIO_PRESETS)."""
    dr = _get_source_module()
    trial_stim, trial_context, trial_block = dr.generate_trial_sequence(
        num_blocks=num_blocks, trials_per_block=trials_per_block, seed=seed)
    params = dr.SCENARIO_PRESETS[scenario]
    data = dr.simulate_dynamic_routing_session(
        num_neurons=num_neurons, trial_stim=trial_stim, trial_context=trial_context,
        bin_size_s=bin_size_s, seed=seed + 1, **params,
    )
    data["trial_block"] = np.array(trial_block)
    return data


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
# 2. Spikes -> smoothed single-trial firing rates
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
# 3. Neural subspace: PCA on condition-averaged (denoised) PSTHs
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

        # PCA via SVD (no sklearn dependency, consistent with parent script)
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
# 4. Simple task-axis (choice axis): mean-difference direction, a
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


# ==========================================================================
# 5. Plotting helpers
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
    (mismatched context, go response = false alarm). Used for both
    aud_target ('aud1') and vis_target ('vis1')."""
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
# 6. Main analysis pipeline
# ==========================================================================

def run_analysis(scenario="clear", n_components=3, out_dir="/tmp",
                  num_blocks=16, trials_per_block=50, num_neurons=16,
                  bin_size_s=0.02, min_trials_per_condition=5, seed=0):
    print(f"Simulating '{scenario}' scenario session "
          f"({num_blocks} blocks x {trials_per_block} trials)...")
    data = simulate_session(scenario=scenario, num_blocks=num_blocks,
                              trials_per_block=trials_per_block,
                              num_neurons=num_neurons, bin_size_s=bin_size_s,
                              seed=seed)

    stim_id = data["stim_id"]
    context = data["context"]
    outcome = label_outcomes(data)
    time_bins = data["time_bins"]

    print("Converting spikes to smoothed single-trial firing rates...")
    rates = spikes_to_rates(data["spike_counts"], bin_size_s=bin_size_s,
                              smooth_sigma_bins=2.0)
    # rates: (n_trials, n_time, n_neurons)

    # ---- 6a. Build the subspace from condition-averaged (denoised) PSTHs
    # Conditions = (stim_id, context, outcome) combinations with enough
    # trials, so the subspace captures stimulus-, context-, and choice-
    # related structure rather than being dominated by trial noise.
    condition_keys = list(zip(stim_id.tolist(), context.tolist(), outcome.tolist()))
    cond_avg_rates, cond_keys, cond_counts = build_condition_averages(
        rates, condition_keys, min_trials=min_trials_per_condition)
    print(f"Fitting PCA subspace on {len(cond_keys)} condition-averaged "
          f"PSTHs (n_components={n_components})...")
    subspace = NeuralSubspace(n_components=n_components).fit(cond_avg_rates)
    evr = subspace.explained_variance_ratio_
    print("  variance explained by top PCs:",
          np.round(evr[:n_components], 3), " cumulative:",
          round(evr[:n_components].sum(), 3))

    # ---- 6b. Project every individual single trial into that subspace
    print("Projecting individual single trials into the subspace...")
    pc_trials = subspace.transform(rates)  # (n_trials, n_time, n_components)

    # ---- 6c. Comparison 1: hit / miss / correct_reject / false_alarm
    plot_outcome_trajectories(
        pc_trials, outcome, pcx=0, pcy=1,
        title=f"Single-trial trajectories by outcome ({scenario} scenario)",
        save_path=os.path.join(out_dir, f"subspace_outcomes_{scenario}.png"),
    )

    plot_scree(subspace, save_path=os.path.join(out_dir, f"subspace_scree_{scenario}.png"))

    # ---- 6d. Comparison 2: correct vs. incorrect response to a FIXED
    # physical stimulus, presented in its context-matched ("correct") block
    # vs. its context-mismatched ("incorrect") block. Done for both
    # aud_target ("aud1") and vis_target ("vis1").
    is_aud1 = (stim_id == "aud_target")
    correct_aud1 = np.where(is_aud1 & (context == "auditory") & (outcome == "hit"))[0]
    incorrect_aud1 = np.where(is_aud1 & (context == "visual") & (outcome == "false_alarm"))[0]
    print(f"aud1 correct (auditory block, hit): n={len(correct_aud1)}   "
          f"aud1 incorrect (visual block, false alarm): n={len(incorrect_aud1)}")

    if len(correct_aud1) >= 2 and len(incorrect_aud1) >= 2:
        plot_stim_correct_vs_incorrect(
            pc_trials, correct_aud1, incorrect_aud1, time_bins,
            stim_label="aud1",
            correct_label="Correct aud1 (auditory block, hit)",
            incorrect_label="Incorrect aud1 (visual block, false alarm)",
            pcx=0, pcy=1,
            title=f"Correct vs. incorrect aud1 (aud_target) response ({scenario} scenario)",
            save_path=os.path.join(out_dir, f"subspace_aud1_correct_vs_incorrect_{scenario}.png"),
        )
    else:
        print("  Not enough trials in one of the aud1 groups to plot "
              "(try more blocks/trials_per_block, or lower "
              "min_trials_per_condition).")

    is_vis1 = (stim_id == "vis_target")
    correct_vis1 = np.where(is_vis1 & (context == "visual") & (outcome == "hit"))[0]
    incorrect_vis1 = np.where(is_vis1 & (context == "auditory") & (outcome == "false_alarm"))[0]
    print(f"vis1 correct (visual block, hit): n={len(correct_vis1)}   "
          f"vis1 incorrect (auditory block, false alarm): n={len(incorrect_vis1)}")

    if len(correct_vis1) >= 2 and len(incorrect_vis1) >= 2:
        plot_stim_correct_vs_incorrect(
            pc_trials, correct_vis1, incorrect_vis1, time_bins,
            stim_label="vis1",
            correct_label="Correct vis1 (visual block, hit)",
            incorrect_label="Incorrect vis1 (auditory block, false alarm)",
            pcx=0, pcy=1,
            title=f"Correct vs. incorrect vis1 (vis_target) response ({scenario} scenario)",
            save_path=os.path.join(out_dir, f"subspace_vis1_correct_vs_incorrect_{scenario}.png"),
        )
    else:
        print("  Not enough trials in one of the vis1 groups to plot "
              "(try more blocks/trials_per_block, or lower "
              "min_trials_per_condition).")

    # ---- 6e. Bonus: mean-difference "choice axis" (lick vs. no-lick),
    # a lightweight analogue of Mante's regression-based targeted axes,
    # projected and shown as a 1-D trace per outcome over time.
    choice_axis = mean_difference_axis(
        rates, data["licked"], ~data["licked"], subspace.mean_, subspace.std_)
    z_rates = (rates - subspace.mean_) / subspace.std_
    choice_proj = z_rates @ choice_axis  # (n_trials, n_time)
    plot_choice_axis(
        pc_trials, choice_proj, outcome, time_bins,
        title=f"Projection onto lick/no-lick 'choice axis' ({scenario} scenario)",
        save_path=os.path.join(out_dir, f"subspace_choice_axis_{scenario}.png"),
    )

    return dict(data=data, rates=rates, subspace=subspace, pc_trials=pc_trials,
                 outcome=outcome, correct_aud1=correct_aud1,
                 incorrect_aud1=incorrect_aud1, correct_vis1=correct_vis1,
                 incorrect_vis1=incorrect_vis1, choice_proj=choice_proj)


if __name__ == "__main__":
    out_dir = "/tmp"
    os.makedirs(out_dir, exist_ok=True)

    results_clear = run_analysis(scenario="clear", out_dir=out_dir, seed=0)

    print("\nRepeating on the 'ambiguous' scenario for comparison "
          "(weaker/graded decision state, weaker context gating)...")
    results_ambiguous = run_analysis(scenario="ambiguous", out_dir=out_dir, seed=0)

    print(f"\nDone. Figures written to {out_dir}/subspace_*.png")
