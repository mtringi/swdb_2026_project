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
from scipy.linalg import eigh


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
# 1b. Demixed PCA (dPCA): separate axes per task marginalization
# ==========================================================================
#
# Ordinary PCA (section 1 above) finds axes that capture the most variance
# overall, mixing "how activity changes over time" and "how activity
# differs between conditions" into the same components. Demixed PCA (Kobak
# et al. 2016, eLife, https://elifesciences.org/articles/10989) instead
# splits the (condition x time) trial-averaged data into three additive
# pieces before doing any dimensionality reduction:
#
#   X_t   "pure time"        -- the time course shared by every condition
#                                (condition-collapsed average profile)
#   X_c   "pure condition"   -- how each condition's overall level differs,
#                                with no time dependence
#   X_ct  "condition x time" -- everything left over: dynamics that differ
#                                BETWEEN conditions (usually the interesting
#                                one -- e.g. "when does the population
#                                diverge between hit and miss")
#
# with X_t + X_c + X_ct = the (mean-subtracted) condition-averaged data.
# This decomposition is exact/balanced here because every condition shares
# the same T time bins (a complete condition x time grid), which is exactly
# `cond_avg_rates` as already computed above for the ordinary PCA subspace.
#
# A SEPARATE set of principal-component-like axes is fit per marginalization
# phi, using the regularized generalized-eigenvalue solution from the paper
# (Methods, eqs. 5-8): decoder axes solve
#
#     C_phi @ w = gamma * (C + lambda * I) @ w
#
# where C_phi is the (neuron x neuron) covariance of that marginalization
# and C is the covariance of the full (unmarginalized, but still
# condition-averaged) data -- this uses the paper's balanced-design
# simplification that marginalizations are mutually orthogonal, so the
# cross-covariance between a marginalization and the full data collapses to
# C_phi. lambda is a ridge penalty, both to keep C invertible (there are
# usually more neurons than condition x time samples) and to keep the
# solution from overfitting a handful of noisy trials; it's chosen by
# cross-validating over single trials (`_choose_regularizer`), mirroring the
# paper's procedure. Single trials are then projected into each
# marginalization's axes exactly like ordinary PCA's `.transform()`, so the
# SAME plotting functions used for the PCA subspace below (outcome
# trajectories, correct-vs-incorrect distance, ...) can be reused unchanged
# on the dPCA 'ct' projection -- the direct analogue of the mixed PCA
# subspace, but with the condition-independent time course factored out.


def marginalize_condition_time(cond_avg_rates):
    """cond_avg_rates: (n_conditions, n_time, n_neurons), already trial-
    averaged per condition (the same object `build_condition_averages`
    produces). Returns (X_t, X_c, X_ct, grand_mean), each of shape
    (n_conditions, n_time, n_neurons) except grand_mean ((n_neurons,)),
    decomposing cond_avg_rates - grand_mean == X_t + X_c + X_ct."""
    grand_mean = cond_avg_rates.mean(axis=(0, 1), keepdims=True)
    centered = cond_avg_rates - grand_mean
    x_t = centered.mean(axis=0, keepdims=True) * np.ones_like(centered)
    x_c = centered.mean(axis=1, keepdims=True) * np.ones_like(centered)
    x_ct = centered - x_t - x_c
    return x_t, x_c, x_ct, grand_mean.reshape(-1)


def _group_indices_by_condition(condition_keys):
    keys = [tuple(k) if isinstance(k, (tuple, list)) else k for k in condition_keys]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    return {k: np.array(v) for k, v in groups.items()}


def _solve_dpca(C_phi, C, lam, n_components):
    """Solve the regularized dPCA generalized eigenproblem
    C_phi w = gamma (C + lam I) w for one marginalization. Returns the top
    `n_components` decoder axes D (n_neurons, n_components) and the
    fraction of TOTAL (unmarginalized) variance each axis captures
    (D_k^T C D_k / trace(C)) -- comparable across marginalizations, so it
    can be shown as grouped bars like a scree plot. Only the decoder is
    kept (not the paper's separate reconstruction encoder), since the only
    thing used downstream is projecting single trials for trajectory plots."""
    n = C.shape[0]
    eigvals, eigvecs = eigh(C_phi, C + lam * np.eye(n))
    order = np.argsort(eigvals)[::-1][:n_components]
    D = eigvecs[:, order]
    # scipy.linalg.eigh(A, B) normalizes eigenvectors so v^T B v == 1, an
    # arbitrary (and lambda-dependent, since B = C + lam*I) scale unrelated
    # to how much variance a direction actually captures -- rescale to unit
    # Euclidean norm so the "explained variance" below, and single-trial
    # projection magnitudes from `.transform()`, are meaningful and
    # comparable across marginalizations/components.
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    total_var = np.trace(C)
    explained = np.array([
        (D[:, k] @ C @ D[:, k]) / total_var if total_var > 0 else 0.0
        for k in range(D.shape[1])
    ])
    return D, explained


def _choose_regularizer(rates, condition_keys, marg, regularizer_grid, mean_, std_,
                          min_trials=5, n_repeats=5, train_frac=0.75, seed=0):
    """Cross-validate the ridge penalty lambda for one marginalization:
    repeatedly split each condition's single trials into a train/test half,
    average each half separately into train/test condition PSTHs, solve the
    dPCA eigenproblem from the TRAIN half's covariances at each candidate
    lambda, and score it by how much of the HELD-OUT test half's
    marginalized variance those axes capture. Picks the lambda that
    generalizes to unseen trials, rather than the lambda->0 solution that
    best (over)fits the training covariance alone -- mirrors the
    cross-validation in Kobak et al. 2016 (Methods)."""
    rng = np.random.default_rng(seed)
    n_neurons = rates.shape[-1]
    groups = _group_indices_by_condition(condition_keys)
    usable = {k: idx for k, idx in groups.items() if len(idx) >= max(2 * min_trials, 4)}
    if len(usable) < 2:
        return regularizer_grid[len(regularizer_grid) // 4]

    def _cov(tensor):
        m = tensor.reshape(-1, n_neurons)
        return (m.T @ m) / max(m.shape[0] - 1, 1)

    scores = np.zeros(len(regularizer_grid))
    n_valid_repeats = 0

    for _ in range(n_repeats):
        train_avgs, test_avgs = [], []
        for idx in usable.values():
            perm = rng.permutation(idx)
            n_train = int(round(len(perm) * train_frac))
            n_train = min(max(n_train, min_trials), len(perm) - min_trials)
            train_avgs.append(rates[perm[:n_train]].mean(axis=0))
            test_avgs.append(rates[perm[n_train:]].mean(axis=0))

        train = (np.stack(train_avgs) - mean_) / std_
        test = (np.stack(test_avgs) - mean_) / std_

        x_t_tr, x_c_tr, x_ct_tr, _ = marginalize_condition_time(train)
        marg_train = dict(t=x_t_tr, c=x_c_tr, ct=x_ct_tr)[marg]
        centered_train = train - train.mean(axis=(0, 1), keepdims=True)

        x_t_te, x_c_te, x_ct_te, _ = marginalize_condition_time(test)
        marg_test_flat = dict(t=x_t_te, c=x_c_te, ct=x_ct_te)[marg].reshape(-1, n_neurons)

        C_train = _cov(centered_train)
        C_phi_train = _cov(marg_train)
        n_valid_repeats += 1

        for gi, lam in enumerate(regularizer_grid):
            eigvals, eigvecs = eigh(C_phi_train, C_train + lam * np.eye(n_neurons))
            top = eigvecs[:, np.argsort(eigvals)[::-1][:3]]
            # same rescaling fix as in `_solve_dpca`: without it, larger
            # lambda -> larger B -> smaller-normed eigenvectors, which
            # would bias this score toward small lambda regardless of
            # actual generalization.
            top = top / np.linalg.norm(top, axis=0, keepdims=True)
            proj = marg_test_flat @ top
            total_var = np.sum(marg_test_flat ** 2) + 1e-12
            scores[gi] += np.sum(proj ** 2) / total_var

    if n_valid_repeats == 0:
        return regularizer_grid[len(regularizer_grid) // 4]
    return regularizer_grid[np.argmax(scores)]


class DemixedSubspace:
    """Regularized demixed PCA (dPCA) over the factors 'condition' and
    'time' (Kobak et al. 2016, eLife) -- see the section comment above for
    the marginalization math and rationale."""

    MARGINALIZATIONS = ("t", "c", "ct")

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.decoders_ = {}                    # marg -> (n_neurons, n_components)
        self.explained_variance_ratio_ = {}    # marg -> (n_components,)
        self.regularizer_ = {}                 # marg -> chosen lambda

    def fit(self, cond_avg_rates, rates_for_cv=None, condition_keys_for_cv=None,
            min_trials_for_cv=5, regularizer_grid=None, n_cv_repeats=5,
            cv_train_frac=0.75, seed=0):
        """cond_avg_rates: (n_conditions, n_time, n_neurons) -- the same
        object `build_condition_averages` returns for the ordinary PCA
        subspace above.

        rates_for_cv / condition_keys_for_cv: OPTIONAL single-trial data
        (rates: (n_trials, n_time, n_neurons); condition_keys: per-trial
        labels, same convention as `build_condition_averages`) used only to
        cross-validate the ridge regularizer lambda, as in the paper. If
        omitted, a small fixed lambda from `regularizer_grid` is used
        instead -- still numerically stable, just not cross-validated."""
        n_cond, n_time, n_neurons = cond_avg_rates.shape
        flat = cond_avg_rates.reshape(-1, n_neurons)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        z = (cond_avg_rates - self.mean_) / self.std_

        x_t, x_c, x_ct, _ = marginalize_condition_time(z)
        centered = z - z.mean(axis=(0, 1), keepdims=True)
        marg_data = dict(t=x_t, c=x_c, ct=x_ct)

        def _cov(tensor):
            m = tensor.reshape(-1, n_neurons)
            return (m.T @ m) / max(m.shape[0] - 1, 1)

        C = _cov(centered)

        if regularizer_grid is None:
            diag_scale = np.trace(C) / n_neurons
            regularizer_grid = diag_scale * np.logspace(-4, 2, 15)

        for marg in self.MARGINALIZATIONS:
            C_phi = _cov(marg_data[marg])
            if rates_for_cv is not None and condition_keys_for_cv is not None:
                lam = _choose_regularizer(
                    rates_for_cv, condition_keys_for_cv, marg, regularizer_grid,
                    self.mean_, self.std_, min_trials=min_trials_for_cv,
                    n_repeats=n_cv_repeats, train_frac=cv_train_frac, seed=seed)
            else:
                lam = regularizer_grid[len(regularizer_grid) // 4]
            self.regularizer_[marg] = lam
            D, evr = _solve_dpca(C_phi, C, lam, self.n_components)
            self.decoders_[marg] = D
            self.explained_variance_ratio_[marg] = evr
        return self

    def transform(self, rates, marginalization):
        """rates: (..., n_neurons) -> (..., n_components) for ONE
        marginalization ('t', 'c', or 'ct')."""
        z = (rates - self.mean_) / self.std_
        return z @ self.decoders_[marginalization]


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


def plot_dpca_variance_explained(dsubspace, save_path=None):
    """Grouped-bar analogue of `plot_scree` for dPCA: variance explained
    (as a fraction of TOTAL population variance, so comparable across
    marginalizations) by each component, grouped by marginalization."""
    margs = list(dsubspace.MARGINALIZATIONS)
    marg_labels = {"t": "pure time\n(condition-independent)",
                    "c": "pure condition\n(time-independent)",
                    "ct": "condition x time\n(interaction)"}
    marg_colors = {"t": "#999999", "c": "#4c72b0", "ct": "#c44e52"}
    n_comp = dsubspace.n_components

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(margs)
    x = np.arange(n_comp)
    for i, m in enumerate(margs):
        evr = dsubspace.explained_variance_ratio_[m]
        ax.bar(x + i * width, evr, width=width, color=marg_colors.get(m),
                label=marg_labels.get(m, m))
    ax.set_xticks(x + width * (len(margs) - 1) / 2)
    ax.set_xticklabels([f"comp {i + 1}" for i in range(n_comp)])
    ax.set_ylabel("Fraction of total population variance")
    ax.set_title("dPCA variance explained by marginalization")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_dpca_component_timecourses(dsubspace, cond_avg_rates, cond_keys, time_bins,
                                       n_components=3, save_path=None):
    """Canonical dPCA figure: rows = marginalization, columns = component
    rank; each panel shows that component's trial-averaged (denoised) time
    course for every kept condition, colored by outcome when the condition
    keys carry one (as `(stim_id, context, outcome)` tuples -- see how
    `condition_keys` is built in `run_real_data_analysis`)."""
    margs = list(dsubspace.MARGINALIZATIONS)
    marg_titles = {"t": "pure time", "c": "pure condition", "ct": "condition x time"}
    fig, axes = plt.subplots(len(margs), n_components,
                               figsize=(3.6 * n_components, 3.0 * len(margs)),
                               squeeze=False)

    def _color_for(key):
        outcome_label = key[-1] if isinstance(key, tuple) and len(key) >= 1 else None
        return OUTCOME_COLORS.get(outcome_label, "#888888")

    for ri, m in enumerate(margs):
        proj = dsubspace.transform(cond_avg_rates, m)  # (n_cond, n_time, n_comp)
        for ci in range(n_components):
            ax = axes[ri][ci]
            for k_idx, key in enumerate(cond_keys):
                ax.plot(time_bins, proj[k_idx, :, ci], color=_color_for(key),
                         lw=1.2, alpha=0.8)
            ax.axhline(0, color="grey", lw=0.5, zorder=0)
            if ri == 0:
                ax.set_title(f"component {ci + 1}")
            if ci == 0:
                ax.set_ylabel(marg_titles[m])
            if ri == len(margs) - 1:
                ax.set_xlabel("Time in trial (s)")

    legend_elems = [Line2D([0], [0], color=c, lw=2, label=OUTCOME_LABELS[k])
                     for k, c in OUTCOME_COLORS.items()]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=8)
    fig.suptitle("dPCA component time courses by marginalization")
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_reward_vs_outcome_comparison(dsubspace, rates, trial_df, outcome, time_bins,
                                        out_dir, noncontingent_col="is_noncontingent_reward",
                                        min_noncontingent=2):
    """Tests whether a dPCA condition x time (interaction) component that
    differs by behavioral outcome (e.g. a late ramp specific to hit trials)
    tracks REWARD specifically, rather than task/decision correctness or the
    licking motor act itself. Hit trials are correct, rewarded, AND licked,
    so on their own they can't distinguish those three explanations.
    Noncontingent ("free") reward trials break that confound: they are
    rewarded without any correct discrimination being made (and typically
    without a lick at all). This compares:

      Hit               -- correct, rewarded, licked
      Noncontingent      -- rewarded, NOT a correct discrimination, no lick
      Correct reject     -- NOT rewarded, no lick

    If noncontingent trials pattern with Hits, that points to reward as the
    driver. If they instead pattern with correct-reject, the signal more
    likely tracks task correctness (or the lick itself, if hits were driving
    it). Skips (with a printed note) if the trial table doesn't have a
    noncontingent-reward column or too few such trials exist."""
    if noncontingent_col not in trial_df.columns:
        print(f"NOTE: '{noncontingent_col}' not in trial table -- skipping "
              f"reward-vs-outcome comparison.")
        return None
    noncontingent = trial_df[noncontingent_col].to_numpy().astype(bool)
    n_nc = int(noncontingent.sum())
    if n_nc < min_noncontingent:
        print(f"NOTE: only {n_nc} noncontingent-reward trials (need >= "
              f"{min_noncontingent}) -- skipping reward-vs-outcome comparison.")
        return None

    hit_idx = np.where(outcome == "hit")[0]
    cr_idx = np.where(outcome == "correct_reject")[0]
    nc_idx = np.where(noncontingent)[0]
    print(f"Reward-vs-outcome comparison: hit n={len(hit_idx)}, "
          f"noncontingent-reward n={len(nc_idx)}, correct-reject n={len(cr_idx)}")

    dpca_trials_ct = dsubspace.transform(rates, "ct")
    n_comp = dsubspace.n_components

    groups = [("Hit (correct, rewarded, licked)", hit_idx, "#1b9e77"),
              ("Noncontingent reward (rewarded, no discrimination)", nc_idx, "#e6ab02"),
              ("Correct reject (unrewarded, no lick)", cr_idx, "#377eb8")]

    fig, axes = plt.subplots(1, n_comp, figsize=(4.4 * n_comp, 4.4), squeeze=False)
    for ci in range(n_comp):
        ax = axes[0][ci]
        for label, idx, color in groups:
            if len(idx) == 0:
                continue
            vals = dpca_trials_ct[idx, :, ci]
            m = vals.mean(axis=0)
            sem = vals.std(axis=0, ddof=1) / np.sqrt(len(idx)) if len(idx) > 1 else np.zeros_like(m)
            ax.plot(time_bins, m, color=color, lw=2.0, label=f"{label} (n={len(idx)})")
            ax.fill_between(time_bins, m - sem, m + sem, color=color, alpha=0.2)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"component {ci + 1}")
        ax.set_xlabel("Time in trial (s)")
        if ci == 0:
            ax.set_ylabel("dPCA condition x time projection (a.u.)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
    fig.suptitle("Reward vs. behavioral-outcome comparison (dPCA condition x time)", y=1.2)
    fig.tight_layout()
    timecourse_path = os.path.join(out_dir, "real_dpca_reward_vs_outcome_timecourses.png")
    fig.savefig(timecourse_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_trajectory_distance(
        dpca_trials_ct, hit_idx, nc_idx, time_bins,
        stim_label="", correct_label="Hit", incorrect_label="Noncontingent reward",
        n_pcs=n_comp,
        title="Hit vs. noncontingent-reward trajectory separation (dPCA ct, all components)",
        save_path=os.path.join(out_dir, "real_dpca_hit_vs_noncontingent_distance.png"),
    )

    if len(cr_idx) >= 2:
        plot_trajectory_distance(
            dpca_trials_ct, nc_idx, cr_idx, time_bins,
            stim_label="", correct_label="Noncontingent reward", incorrect_label="Correct reject",
            n_pcs=n_comp,
            title="Noncontingent-reward vs. correct-reject trajectory separation (dPCA ct)",
            save_path=os.path.join(out_dir, "real_dpca_noncontingent_vs_correct_reject_distance.png"),
        )

    print(f"Reward-vs-outcome figures written to {out_dir}/real_dpca_reward_vs_outcome_*.png "
          f"and {out_dir}/real_dpca_*_distance.png")
    return dict(dpca_trials_ct=dpca_trials_ct, hit_idx=hit_idx, nc_idx=nc_idx, cr_idx=cr_idx)


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

    # --- dPCA (demixed PCA; see section 1b) ---
    run_dpca=True,             # set False to skip dPCA and only run ordinary PCA
    dpca_n_components=3,
    dpca_cv_repeats=5,         # train/test splits used to cross-validate the ridge penalty
    dpca_cv_train_frac=0.75,
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

    # ---- 8e. Demixed PCA (dPCA): same condition-averaged PSTHs as the PCA
    # subspace above, but split into separate 'pure time' / 'pure condition'
    # / 'condition x time' axes first (section 1b) so dynamics shared by
    # every condition don't crowd out the condition-DEPENDENT dynamics in
    # the same components, the way they can in ordinary PCA.
    dsubspace = None
    if cfg.get("run_dpca", True):
        print("Fitting demixed PCA (dPCA) on the same condition-averaged PSTHs...")
        dpca_n = cfg.get("dpca_n_components", 3)
        dsubspace = DemixedSubspace(n_components=dpca_n).fit(
            cond_avg_rates, rates_for_cv=rates, condition_keys_for_cv=condition_keys,
            min_trials_for_cv=cfg["min_trials_per_condition"],
            n_cv_repeats=cfg.get("dpca_cv_repeats", 5),
            cv_train_frac=cfg.get("dpca_cv_train_frac", 0.75),
        )
        for m in dsubspace.MARGINALIZATIONS:
            print(f"  [{m}] regularizer={dsubspace.regularizer_[m]:.4g}  "
                  f"variance explained={np.round(dsubspace.explained_variance_ratio_[m], 3)}")

        plot_dpca_variance_explained(
            dsubspace, save_path=os.path.join(out_dir, "real_dpca_variance_explained.png"))
        plot_dpca_component_timecourses(
            dsubspace, cond_avg_rates, cond_keys, time_bins, n_components=dpca_n,
            save_path=os.path.join(out_dir, "real_dpca_component_timecourses.png"))

        # Direct analogues of the PCA outcome-trajectory / correct-vs-
        # incorrect figures above (8b/8c), built from the demixed
        # 'condition x time' component instead of ordinary mixed PCs --
        # reuses the exact same plotting functions since they only require
        # a (n_trials, n_time, n_components) projection array. The
        # secondary comparisons (8c-2, 8c-3) can be produced the same way
        # by swapping pc_trials for dpca_trials_ct.
        dpca_trials_ct = dsubspace.transform(rates, "ct")

        plot_outcome_trajectories(
            dpca_trials_ct, outcome, pcx=0, pcy=1,
            title="Single-trial trajectories by outcome (dPCA condition x time component)",
            save_path=os.path.join(out_dir, "real_dpca_outcomes.png"),
        )

        for stim, modality in unique_stims:
            is_this_stim = (stim_id == stim)
            matched_ctx = is_this_stim & (context == modality) & (outcome == "hit")
            mismatched_ctx = is_this_stim & (context != modality) & (outcome == "false_alarm")
            if len(matched_ctx.nonzero()[0]) >= 2 and len(mismatched_ctx.nonzero()[0]) >= 2:
                correct_idx = np.where(matched_ctx)[0]
                incorrect_idx = np.where(mismatched_ctx)[0]
                plot_stim_correct_vs_incorrect(
                    dpca_trials_ct, correct_idx, incorrect_idx, time_bins,
                    stim_label=str(stim),
                    correct_label=f"Correct {stim} (matched context, hit)",
                    incorrect_label=f"Incorrect {stim} (mismatched context, false alarm)",
                    pcx=0, pcy=1,
                    title=f"Correct vs. incorrect {stim} response, dPCA condition x time component",
                    save_path=os.path.join(
                        out_dir, f"real_dpca_{stim}_correct_vs_incorrect.png"),
                )

        plot_reward_vs_outcome_comparison(
            dsubspace, rates, trial_df, outcome, time_bins, out_dir)

    print(f"Done. Figures written to {out_dir}/real_subspace_*.png"
          + (f" and {out_dir}/real_dpca_*.png" if dsubspace is not None else ""))
    return dict(rates=rates, subspace=subspace, pc_trials=pc_trials,
                 dpca_subspace=dsubspace,
                 dpca_trials_ct=(dsubspace.transform(rates, "ct") if dsubspace is not None else None),
                 stim_id=stim_id, context=context, outcome=outcome,
                 choice_proj=choice_proj, time_bins=time_bins)


if __name__ == "__main__":
    # Edit CONFIG above with your real file paths / column names, then:
    results = run_real_data_analysis(CONFIG)
