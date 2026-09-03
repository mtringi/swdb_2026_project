"""
Per-stimulus Poisson-HMM fitting (pooled across task contexts), a matched
null-data control, and a cross-stimulus generalization test, built on top of
`hmm_helpers.py` (fitting/decoding) and `hmm_null_control.py` (surrogate null
spike-train generation).

Nothing here is a reimplementation of the fitting/decoding math -- every HMM
fit and every posterior decode goes through `hmm_helpers`'s
`initialize_and_fit_poisson_hmm` / `compute_posterior_state_probabilities`,
so results are directly comparable to any other analysis built on that
module.

Three pieces:

1. `run_area_stimulus_analysis` -- for a given units table (e.g. `area1_units`)
   and trial table, fits ONE HMM per stimulus, pooling trials across task
   contexts (`rewarded_modality` blocks) for that stimulus, and fits a
   matched null-control HMM (see `hmm_null_control.simulate_null_spike_trains`)
   alongside it for comparison.

2. `cross_generalize_stimulus_pair` -- fits an HMM on one stimulus's trials
   only, then decodes a DIFFERENT (held-out) stimulus's trials with those
   FIXED parameters via the forward-backward algorithm (no re-fitting), to
   test whether a state transition found for one stimulus is a genuine,
   stimulus-general population dynamic or an artifact of fitting each
   stimulus separately.

3. `trial_outcome_labels` / `plot_trial_state_raster_with_outcomes` /
   `plot_cross_generalization_comparison` -- behavioral-outcome-aware
   plotting on top of (1) and (2).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import mannwhitneyu

from hmm_helpers import (
    initialize_and_fit_poisson_hmm,
    compute_posterior_state_probabilities,
    reorder_states_temporally,
    detect_state_onsets,
)
from hmm_null_control import bin_units_table_to_counts, simulate_null_spike_trains


# ==========================================================================
# 1. Behavioral outcome labeling
# ==========================================================================

OUTCOME_COLORS = {'hit': '#1b9e77', 'correct_reject': '#377eb8', 'incorrect': '#d95f02'}


def trial_outcome_labels(trials_df):
    """Three-way outcome per trial from the Dynamic Routing behavior table's
    precomputed `is_hit` / `is_correct_reject` / `is_incorrect` columns.
    Any trial not flagged by one of these (e.g. a miss that isn't scored
    `is_incorrect`, or a trial with no behavioral read) falls back to
    'incorrect' unless it's a hit or correct reject."""
    n = len(trials_df)
    outcome = np.full(n, 'incorrect', dtype=object)
    outcome[trials_df['is_correct_reject'].fillna(False).to_numpy(dtype=bool)] = 'correct_reject'
    outcome[trials_df['is_hit'].fillna(False).to_numpy(dtype=bool)] = 'hit'
    return outcome


# ==========================================================================
# 2. Bin one stimulus's trials + fit/decode helper
# ==========================================================================

def bin_stim_trials(units_df, trials_df, stim, bin_edges,
                     unit_col='unit_id', spike_col='spike_times', stim_col='stim_name'):
    """Selects `trials_df` rows for one stimulus (pooling across whatever
    `rewarded_modality` context each trial happened to be in -- no context
    filtering here) and bins spikes from `units_df` around `stim_start_time`.

    Returns (counts, stim_trials, unit_ids).
    """
    stim_trials = trials_df[trials_df[stim_col] == stim]
    trial_onsets = stim_trials['stim_start_time'].to_numpy()
    counts, unit_ids = bin_units_table_to_counts(
        units_df, trial_onsets, bin_edges, unit_col=unit_col, spike_col=spike_col)
    return counts, stim_trials, unit_ids


def fit_and_decode(counts, num_states, bin_centers, bin_width,
                    n_restarts=5, num_iters=60, seed=0):
    """Fits a Poisson HMM (`hmm_helpers.initialize_and_fit_poisson_hmm`),
    decodes smoothed posteriors, and temporally reorders states (state 0 =
    earliest-active, last state = latest-active)."""
    hmm, params, ll, lls_trace = initialize_and_fit_poisson_hmm(
        counts, num_states=num_states, bin_width=bin_width,
        n_restarts=n_restarts, num_iters=num_iters, seed=seed)
    probs = compute_posterior_state_probabilities(hmm, params, counts)
    reordered_probs, reordered_transitions, reordered_rates, order = reorder_states_temporally(
        probs, params, bin_centers)
    return dict(hmm=hmm, params=params, ll=ll, lls_trace=lls_trace,
                probs=reordered_probs, transitions=reordered_transitions,
                rates_hz=reordered_rates, order=order)


# ==========================================================================
# 3. Per-stimulus fit (pooled across contexts) + matched null control
# ==========================================================================

def run_area_stimulus_analysis(units_df, trials_df, area_label, bin_edges, num_states=4,
                                 n_restarts=5, num_iters=60, seed=0, stim_list=None,
                                 min_trials=10, unit_col='unit_id', spike_col='spike_times',
                                 stim_col='stim_name'):
    """For `units_df` (e.g. `area1_units`), fits ONE HMM per stimulus in
    `stim_list` (default: every non-catch stimulus present), pooling trials
    across task contexts, and fits a matched null-control HMM
    (`hmm_null_control.simulate_null_spike_trains`) on surrogate data with
    the same firing rates / trial count / trial-to-trial variability but no
    true state change, for direct comparison.

    Returns {stim: {counts, unit_ids, stim_trials, outcome_labels, real,
    null, real_onsets, null_onsets, bin_centers}}, where `real`/`null` are
    the dicts returned by `fit_and_decode`, and `real_onsets`/`null_onsets`
    are onset latencies (seconds) of the last (latest) state.
    """
    bin_width = float(np.diff(bin_edges).mean())
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    trials_df = trials_df.loc[~trials_df['is_catch'].fillna(False)]
    if stim_list is None:
        stim_list = sorted(trials_df[stim_col].dropna().unique().tolist())

    late_state = num_states - 1
    results = {}
    for stim in stim_list:
        print(f'\n=== {area_label}: stimulus "{stim}" (pooled across contexts) ===')
        counts, stim_trials, unit_ids = bin_stim_trials(
            units_df, trials_df, stim, bin_edges, unit_col=unit_col,
            spike_col=spike_col, stim_col=stim_col)
        print(f'  {counts.shape[0]} trials, {counts.shape[2]} units')
        if counts.shape[0] < min_trials:
            print(f'  skipping: fewer than {min_trials} trials')
            continue

        real_fit = fit_and_decode(counts, num_states, bin_centers, bin_width,
                                   n_restarts=n_restarts, num_iters=num_iters, seed=seed)
        print(f'  real fit LL = {real_fit["ll"]:.1f}')

        null_counts, _, _, rate_profile = simulate_null_spike_trains(
            units_df, stim_trials['stim_start_time'].to_numpy(), bin_edges,
            unit_col=unit_col, spike_col=spike_col, seed=seed + 1)
        null_fit = fit_and_decode(null_counts, num_states, bin_centers, bin_width,
                                   n_restarts=n_restarts, num_iters=num_iters, seed=seed + 1)
        print(f'  null-control fit LL = {null_fit["ll"]:.1f}')

        outcome_labels = trial_outcome_labels(stim_trials)
        real_onsets = detect_state_onsets(real_fit['probs'], bin_centers, late_state)['onset_time_s'].to_numpy()
        null_onsets = detect_state_onsets(null_fit['probs'], bin_centers, late_state)['onset_time_s'].to_numpy()

        results[stim] = dict(
            counts=counts, null_counts=null_counts, unit_ids=unit_ids, stim_trials=stim_trials,
            outcome_labels=outcome_labels, real=real_fit, null=null_fit,
            real_onsets=real_onsets, null_onsets=null_onsets, bin_centers=bin_centers,
        )
    return results


# ==========================================================================
# 4. Cross-stimulus generalization test
# ==========================================================================

def cross_generalize_stimulus_pair(units_df, trials_df, stim_a, stim_b, bin_edges, num_states=4,
                                     n_restarts=5, num_iters=60, seed=0,
                                     unit_col='unit_id', spike_col='spike_times', stim_col='stim_name'):
    """Fits an HMM on `stim_a` trials ONLY, then decodes `stim_b`'s held-out
    trials with those FIXED parameters via the forward-backward algorithm
    (`hmm_helpers.compute_posterior_state_probabilities` -- no re-fitting).
    Also fits `stim_b` natively (its own HMM) for reference.

    A coherent, similarly-timed transition recovered on `stim_b` under the
    fixed `stim_a` parameters indicates a stimulus-general population
    dynamic; a latency shift or a collapse in how many trials ever cross
    threshold indicates the `stim_a` fit was stimulus-specific.

    Returns dict(fit_a, fit_b_native, cross_probs, native_a_onsets,
    cross_b_onsets, native_b_onsets, bin_centers, trials_a, trials_b).
    """
    bin_width = float(np.diff(bin_edges).mean())
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    late_state = num_states - 1

    counts_a, trials_a, _ = bin_stim_trials(units_df, trials_df, stim_a, bin_edges,
                                              unit_col=unit_col, spike_col=spike_col, stim_col=stim_col)
    counts_b, trials_b, _ = bin_stim_trials(units_df, trials_df, stim_b, bin_edges,
                                              unit_col=unit_col, spike_col=spike_col, stim_col=stim_col)

    fit_a = fit_and_decode(counts_a, num_states, bin_centers, bin_width,
                            n_restarts=n_restarts, num_iters=num_iters, seed=seed)

    # forward-backward ONLY, using stim_a's fixed fitted parameters, applied
    # to stim_b's held-out trials -- no fit_em call here.
    cross_probs_raw = compute_posterior_state_probabilities(fit_a['hmm'], fit_a['params'], counts_b)
    cross_probs = cross_probs_raw[:, :, fit_a['order']]  # same state ordering as fit_a

    fit_b_native = fit_and_decode(counts_b, num_states, bin_centers, bin_width,
                                   n_restarts=n_restarts, num_iters=num_iters, seed=seed + 1)

    native_a_onsets = detect_state_onsets(fit_a['probs'], bin_centers, late_state)['onset_time_s'].to_numpy()
    cross_b_onsets = detect_state_onsets(cross_probs, bin_centers, late_state)['onset_time_s'].to_numpy()
    native_b_onsets = detect_state_onsets(fit_b_native['probs'], bin_centers, late_state)['onset_time_s'].to_numpy()

    return dict(fit_a=fit_a, fit_b_native=fit_b_native, cross_probs=cross_probs,
                native_a_onsets=native_a_onsets, cross_b_onsets=cross_b_onsets,
                native_b_onsets=native_b_onsets, bin_centers=bin_centers,
                trials_a=trials_a, trials_b=trials_b)


# ==========================================================================
# 5. Plotting
# ==========================================================================

def plot_trial_state_raster_with_outcomes(state_probs, outcome_labels, bin_centers, num_states,
                                            title=None, outcome_colors=None, ax=None):
    """Dominant-state raster (rows = trials) with a colored strip alongside
    each row for the trial's behavioral outcome ('hit' / 'correct_reject' /
    'incorrect' by default), plus a legend. Trials are sorted by outcome,
    then by the onset latency of the last (latest) state.

    Parameters
    ----------
    state_probs : np.ndarray, shape (num_trials, num_timesteps, num_states)
    outcome_labels : array-like, shape (num_trials,) -- from `trial_outcome_labels`.
    bin_centers : np.ndarray, shape (num_timesteps,), seconds relative to trial onset.
    """
    if outcome_colors is None:
        outcome_colors = OUTCOME_COLORS

    outcome_labels = np.asarray(outcome_labels, dtype=object)
    times_ms = bin_centers * 1000
    num_trials = state_probs.shape[0]
    dominant_state = np.argmax(state_probs, axis=-1)
    late_state = num_states - 1

    onset_bin = np.array([
        np.argmax(dominant_state[t] == late_state) if (dominant_state[t] == late_state).any()
        else len(bin_centers)
        for t in range(num_trials)
    ])
    outcome_rank = np.array([
        list(outcome_colors).index(label) if label in outcome_colors else len(outcome_colors)
        for label in outcome_labels
    ])
    order = np.lexsort((onset_bin, outcome_rank))

    palette = sns.color_palette('tab10', num_states)
    cmap_discrete = mcolors.ListedColormap(palette)
    bounds = np.arange(num_states + 1) - 0.5
    norm_discrete = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.05 * num_trials)), dpi=150)
    else:
        fig = ax.figure

    im = ax.imshow(
        dominant_state[order], aspect='auto', cmap=cmap_discrete, norm=norm_discrete,
        extent=[times_ms[0], times_ms[-1], num_trials, 0], interpolation='nearest',
    )
    ax.axvline(0, color='white', ls='--', lw=1.5)
    ax.set_xlabel('Time re: stimulus onset (ms)')
    ax.set_ylabel('Trials (sorted by outcome, then state onset)')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, pad=0.10, ticks=range(num_states))
    cbar.ax.set_yticklabels([f'State {s + 1}' for s in range(num_states)], fontsize=8)

    # colored outcome strip just to the left of the raster
    strip_width_ms = 0.03 * (times_ms[-1] - times_ms[0])
    strip_x0 = times_ms[0] - strip_width_ms
    for row, trial_idx in enumerate(order):
        color = outcome_colors.get(outcome_labels[trial_idx], '#999999')
        ax.add_patch(plt.Rectangle((strip_x0, row), strip_width_ms, 1,
                                    facecolor=color, edgecolor='none', clip_on=False))
    ax.set_xlim(strip_x0, times_ms[-1])

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in outcome_colors.values()]
    legend_names = [name.replace('_', ' ').title() for name in outcome_colors.keys()]
    ax.legend(legend_handles, legend_names, loc='upper left', bbox_to_anchor=(1.32, 1),
              fontsize=8, title='Outcome', frameon=True)

    return fig, ax


def plot_cross_generalization_comparison(native_onsets, cross_onsets,
                                            native_name='Native fit',
                                            cross_name='Cross-decoded (fixed params)',
                                            title=None, save_path=None):
    """Compares the onset-latency distribution of one state between a NATIVE
    fit (HMM fit and decoded on the same stimulus's own trials) and a
    CROSS-DECODED condition (HMM fit on a different stimulus's trials,
    those fixed parameters applied to this stimulus's held-out trials via
    forward-backward only -- no re-fitting).

    Returns (fig, axes, stats); `stats` holds each condition's
    trial-crossing fraction and (if both have >1 valid trial) a
    Mann-Whitney U test on the latency shift.
    """
    native_onsets = np.asarray(native_onsets, dtype=float)
    cross_onsets = np.asarray(cross_onsets, dtype=float)
    native_valid = native_onsets[~np.isnan(native_onsets)] * 1000
    cross_valid = cross_onsets[~np.isnan(cross_onsets)] * 1000
    frac_native = len(native_valid) / len(native_onsets) if len(native_onsets) else np.nan
    frac_cross = len(cross_valid) / len(cross_onsets) if len(cross_onsets) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    ax_kde = axes[0]
    for vals, name, color in [(native_valid, native_name, '#1b9e77'),
                               (cross_valid, cross_name, '#d95f02')]:
        if len(vals) > 1:
            sns.kdeplot(vals, ax=ax_kde, color=color, lw=2.2, fill=True, alpha=0.15,
                        label=f'{name} (n={len(vals)}, med={np.median(vals):.0f} ms)')
    ax_kde.axvline(0, color='black', ls='--', lw=1.2, alpha=0.6)
    ax_kde.set_xlabel('Onset latency re: stimulus onset (ms)')
    ax_kde.set_ylabel('Probability density')
    ax_kde.set_title('Onset latency distribution', fontsize=11)
    ax_kde.legend(fontsize=8)

    ax_bar = axes[1]
    bars = ax_bar.bar([native_name, cross_name], [frac_native, frac_cross],
                       color=['#1b9e77', '#d95f02'], alpha=0.85)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel('Fraction of trials crossing threshold')
    ax_bar.set_title('State recovered?', fontsize=11)
    for bar, frac in zip(bars, [frac_native, frac_cross]):
        if not np.isnan(frac):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, frac + 0.02, f'{frac:.2f}',
                        ha='center', fontsize=9)

    stats = dict(frac_native=frac_native, frac_cross=frac_cross)
    if len(native_valid) > 1 and len(cross_valid) > 1:
        stat, p = mannwhitneyu(native_valid, cross_valid)
        stats['mannwhitney_u'] = float(stat)
        stats['mannwhitney_p'] = float(p)
        note = f'Mann-Whitney U p = {p:.3g} (latency shift)'
    else:
        note = 'insufficient trials for a latency-shift test'
    fig.suptitle((title + ' -- ' if title else '') + note, fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig, axes, stats
