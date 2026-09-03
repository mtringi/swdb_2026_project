"""
Null-model control for the Poisson-HMM state-transition pipeline.

The question this answers: if we hand the HMM pipeline data from a process
that is, BY CONSTRUCTION, continuous and single-state (no true discrete
jump anywhere), does it still come back reporting a confident two-state
sequence with a tight, well-defined transition latency? If so, the pipeline
is biased toward discovering discreteness regardless of ground truth, and
any "state" found in real data is suspect until this control fails to find
one.

Pipeline reproduced here verbatim (same fitting/decoding/latency logic as
`hmm_simulation_tests.ipynb`, so this is genuinely "your exact pipeline"
and not a reimplementation):
    fit_poisson_hmm -> decode_trial_state_probs -> find_late_state_onset

What's new here is the null-data generator, `simulate_null_spike_trains`,
and the orchestrator, `run_null_control_test`.

Null generative model
----------------------
For each unit, real spike counts are binned from `units_df` (one row per
unit, full-session `spike_times`) the same way as the rest of this repo's
real-data pipelines. From that binned data we build a SINGLE smooth,
time-varying rate profile per unit (the trial-averaged PSTH, Gaussian-
smoothed to remove any bin-to-bin roughness -- there is no per-trial
latency jitter, no discrete regime, just one fixed continuous function of
time shared by every trial). Trial-to-trial variability is added back in
as a single scalar multiplicative gain per (unit, trial), drawn from a
Gamma distribution whose variance is fit to match the real data's observed
trial-to-trial overdispersion (excess variance over Poisson) -- this is
the standard doubly-stochastic-Poisson / negative-binomial way to get
realistic trial-to-trial rate variability without introducing any
timing-locked discrete event. Null spike counts are then Poisson draws
around `gain * smoothed_rate_profile`.

This matches real firing rates (the profile *is* the real smoothed PSTH),
real trial count (one null trial per real trial), and real trial-to-trial
variability (matched-overdispersion gain) -- while guaranteeing, by
construction, that there is no true state transition anywhere in the data.

Usage
-----
    from hmm_null_control import run_null_control_test

    result = run_null_control_test(
        area1_units, trial_onsets, bin_edges, num_states=2, n_null_iters=10,
    )
    result.summary()
"""

from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from scipy.ndimage import gaussian_filter1d
from dynamax.hidden_markov_model import PoissonHMM


# ==========================================================================
# 1. Bin a units table's spike times into (n_trials, n_bins, n_units) counts
# ==========================================================================

def bin_units_table_to_counts(units_df, trial_onsets, bin_edges, unit_ids=None,
                                unit_col="unit_id", spike_col="spike_times"):
    """`units_df`: one row per unit (e.g. `area1_units`), `spike_col` holding
    that unit's FULL-SESSION spike times in absolute seconds -- same format
    used by `bin_session_spike_times_to_counts` in `neural_subspace_real_data.py`
    and `bin_session_spike_times` in `glm_hmm_auditory.ipynb`. `trial_onsets`:
    per-trial alignment time (s), same absolute time base as `spike_col`.
    `unit_ids`: optional subset/order of `units_df[unit_col]` values to use
    as columns; defaults to every unit in `units_df`, in table order.

    Returns spike_counts: (n_trials, n_bins, n_units) integer array.
    """
    if unit_ids is None:
        unit_ids = units_df[unit_col].to_numpy()

    spikes_by_unit = {
        getattr(row, unit_col): np.sort(np.asarray(getattr(row, spike_col), dtype=float))
        for row in units_df.itertuples(index=False)
    }

    n_trials, n_units, n_bins = len(trial_onsets), len(unit_ids), len(bin_edges) - 1
    counts = np.zeros((n_trials, n_bins, n_units), dtype=int)
    trial_onsets = np.asarray(trial_onsets, dtype=float)

    n_missing = 0
    for ui, uid in enumerate(unit_ids):
        spikes = spikes_by_unit.get(uid)
        if spikes is None or len(spikes) == 0:
            n_missing += 1
            continue
        for ti, onset in enumerate(trial_onsets):
            local_edges = bin_edges + onset
            i0, i1 = np.searchsorted(spikes, [local_edges[0], local_edges[-1]])
            c, _ = np.histogram(spikes[i0:i1], bins=local_edges)
            counts[ti, :, ui] = c

    if n_missing:
        print(f"  (found no spikes for {n_missing} of {n_units} unit_ids)")
    return counts, np.asarray(unit_ids)


# ==========================================================================
# 2. Generate surrogate null spike trains: continuous, single-state,
#    no true jump anywhere -- matched to real rates/trial-count/variability
# ==========================================================================

def simulate_null_spike_trains(units_df, trial_onsets, bin_edges, unit_ids=None,
                                 unit_col="unit_id", spike_col="spike_times",
                                 smoothing_sigma_bins=3.0, n_null_trials=None,
                                 seed=0):
    """Simulate a surrogate spike-train dataset from a known continuous,
    single-state generative process with no true state change anywhere,
    matched to the real data in `units_df` on:
      - firing rate (per-unit smoothed trial-averaged PSTH),
      - trial count (defaults to `len(trial_onsets)`),
      - trial-to-trial variability (per-unit overdispersion, matched via a
        Gamma-distributed per-trial gain).

    Each unit's rate is a SINGLE fixed smooth function of time, identical
    across all trials up to a random per-trial scalar gain -- there is no
    per-trial latency, no regime switch, nothing for an HMM to legitimately
    lock onto beyond a smoothly time-varying Poisson rate.

    Returns
    -------
    null_spike_counts : np.ndarray, shape (n_null_trials, n_bins, n_units)
    time_bins : np.ndarray, shape (n_bins,) -- bin centers relative to
        `trial_onsets` (seconds)
    unit_ids : np.ndarray, shape (n_units,)
    rate_profile : np.ndarray, shape (n_bins, n_units) -- the smoothed
        single-state rate (expected counts/bin) each null trial was drawn
        around, for sanity-checking that it's in fact smooth/continuous
    """
    real_counts, unit_ids = bin_units_table_to_counts(
        units_df, trial_onsets, bin_edges, unit_ids=unit_ids,
        unit_col=unit_col, spike_col=spike_col)
    n_real_trials, n_bins, n_units = real_counts.shape
    n_null_trials = n_real_trials if n_null_trials is None else n_null_trials
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # single smooth, continuous rate profile per unit (trial-averaged PSTH,
    # Gaussian-smoothed so any real sharp/discrete structure is washed out)
    mean_profile = real_counts.mean(axis=0)  # (n_bins, n_units)
    rate_profile = gaussian_filter1d(mean_profile, sigma=smoothing_sigma_bins,
                                      axis=0, mode="nearest")
    rate_profile = np.clip(rate_profile, 1e-4, None)

    # per-unit trial-to-trial overdispersion, matched from real per-trial
    # total counts: Var(N) = mu*E[gain] + mu^2*Var(gain) for a Poisson-Gamma
    # mixture with E[gain] = 1, so Var(gain) = max(0, (var - mu) / mu^2)
    trial_totals = real_counts.sum(axis=1)  # (n_trials, n_units)
    mu = trial_totals.mean(axis=0)
    var = trial_totals.var(axis=0, ddof=1) if n_real_trials > 1 else np.zeros(n_units)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_var = np.where(mu > 0, (var - mu) / np.clip(mu, 1e-8, None) ** 2, 0.0)
    gain_var = np.clip(gain_var, 0.0, None)

    rng = np.random.default_rng(seed)
    null_counts = np.zeros((n_null_trials, n_bins, n_units), dtype=int)
    for ui in range(n_units):
        if gain_var[ui] > 1e-8:
            shape = 1.0 / gain_var[ui]
            gains = rng.gamma(shape=shape, scale=1.0 / shape, size=n_null_trials)
        else:
            gains = np.ones(n_null_trials)
        trial_rates = gains[:, None] * rate_profile[None, :, ui]  # (trials, bins)
        null_counts[:, :, ui] = rng.poisson(trial_rates)

    return null_counts, time_bins, unit_ids, rate_profile


# ==========================================================================
# 3. Exact HMM-fitting pipeline (mirrors hmm_simulation_tests.ipynb)
# ==========================================================================

def fit_poisson_hmm(spike_counts, num_states, num_restarts=10, num_em_iters=100,
                     seed=0, verbose=True):
    """Fit a Poisson HMM via EM with multiple random restarts, keeping the
    highest-log-likelihood fit. `spike_counts`: (n_trials, n_bins, n_units)."""
    emissions = jnp.array(spike_counts)
    num_neurons = spike_counts.shape[-1]

    best_ll = -jnp.inf
    best_params = None
    best_hmm = None

    key = jr.PRNGKey(seed)
    for restart in range(num_restarts):
        key, subkey = jr.split(key)
        hmm = PoissonHMM(num_states, num_neurons)
        params, props = hmm.initialize(subkey)
        params, lls = hmm.fit_em(params, props, emissions, num_iters=num_em_iters,
                                  verbose=False)
        final_ll = lls[-1]
        if verbose:
            print(f"  restart {restart + 1}/{num_restarts}: final LL = {final_ll:.1f}")
        if final_ll > best_ll:
            best_ll = final_ll
            best_params = params
            best_hmm = hmm

    return best_hmm, best_params, float(best_ll)


def decode_trial_state_probs(hmm, params, spike_counts):
    """Per-trial smoothed state-probability trajectories + Viterbi paths.
    Returns state_probs (n_trials, n_bins, n_states), viterbi_paths
    (n_trials, n_bins)."""
    emissions = jnp.array(spike_counts)
    num_trials = spike_counts.shape[0]

    state_probs, viterbi_paths = [], []
    for trial in range(num_trials):
        trial_emissions = emissions[trial]
        posterior = hmm.smoother(params, trial_emissions)
        state_probs.append(np.array(posterior.smoothed_probs))
        viterbi_paths.append(np.array(hmm.most_likely_states(params, trial_emissions)))

    return np.stack(state_probs), np.stack(viterbi_paths)


def find_late_state_onset(state_probs, late_state_idx, time_bins,
                           prob_threshold=0.5, min_time_s=0.1):
    """First time each trial's `late_state_idx` probability crosses
    `prob_threshold`, after `min_time_s`. NaN if it never happens."""
    num_trials = state_probs.shape[0]
    onsets = np.full(num_trials, np.nan)
    for trial in range(num_trials):
        probs = state_probs[trial, :, late_state_idx]
        above = np.where((probs > prob_threshold) & (time_bins > min_time_s))[0]
        if len(above) > 0:
            onsets[trial] = time_bins[above[0]]
    return onsets


def identify_late_state(state_probs, time_bins, late_window=None):
    """Same heuristic as hmm_simulation_tests.ipynb: the state with highest
    mean posterior probability within `late_window` (default: second half
    of the trial) is the "late"/candidate-transition state."""
    if late_window is None:
        late_window = time_bins > np.median(time_bins)
    mean_prob_in_window = state_probs[:, late_window, :].mean(axis=(0, 1))
    return int(np.argmax(mean_prob_in_window))


# ==========================================================================
# 4. Orchestrator: run the exact pipeline on surrogate null data
# ==========================================================================

@dataclass
class NullControlResult:
    num_states: int
    prob_threshold: float
    per_iter: list = field(default_factory=list)  # one dict per null dataset

    def summary(self):
        frac_found = np.array([it["frac_trials_with_onset"] for it in self.per_iter])
        conf = np.array([it["median_confidence"] for it in self.per_iter])
        spread = np.array([it["onset_latency_iqr_s"] for it in self.per_iter])

        print(f"Null control: {len(self.per_iter)} surrogate dataset(s), "
              f"{self.num_states}-state HMM, threshold={self.prob_threshold}")
        print(f"  fraction of trials crossing threshold: "
              f"mean={frac_found.mean():.2f}, range=[{frac_found.min():.2f}, {frac_found.max():.2f}]")
        print(f"  median late-state posterior confidence: "
              f"mean={conf.mean():.2f}, range=[{conf.min():.2f}, {conf.max():.2f}]")
        print(f"  onset-latency IQR (s), tighter = more 'well-defined': "
              f"mean={spread.mean():.3f}, range=[{spread.min():.3f}, {spread.max():.3f}]")

        biased = (frac_found.mean() > 0.5) and (conf.mean() > 0.8)
        if biased:
            print("  VERDICT: pipeline reports confident, well-defined transitions on "
                  "null data with no true state change -- it is biased toward finding "
                  "discreteness, and 'states' found in real data are suspect.")
        else:
            print("  VERDICT: pipeline does not confidently report sharp transitions on "
                  "null data -- this control passes.")
        return dict(biased=biased, frac_trials_with_onset=frac_found,
                    median_confidence=conf, onset_latency_iqr_s=spread)


def run_null_control_test(units_df, trial_onsets, bin_edges, unit_ids=None,
                            unit_col="unit_id", spike_col="spike_times",
                            num_states=2, num_restarts=8, num_em_iters=100,
                            smoothing_sigma_bins=3.0, prob_threshold=0.5,
                            min_time_s=0.1, late_window=None, n_null_iters=10,
                            seed=0, verbose=False):
    """Generate `n_null_iters` independent surrogate null datasets from
    `units_df` (continuous, single-state, no true jump -- see
    `simulate_null_spike_trains`) and run the exact HMM pipeline
    (`fit_poisson_hmm` -> `decode_trial_state_probs` -> `find_late_state_onset`)
    on each. If the pipeline still confidently reports sharp `num_states`-state
    transitions with tight, well-defined latencies on this null data, the
    method is biased toward discovering discreteness regardless of ground
    truth.

    Returns a `NullControlResult`; call `.summary()` to print the verdict.
    """
    result = NullControlResult(num_states=num_states, prob_threshold=prob_threshold)

    for it in range(n_null_iters):
        null_counts, time_bins, resolved_unit_ids, rate_profile = simulate_null_spike_trains(
            units_df, trial_onsets, bin_edges, unit_ids=unit_ids,
            unit_col=unit_col, spike_col=spike_col,
            smoothing_sigma_bins=smoothing_sigma_bins, seed=seed + it)

        hmm, params, ll = fit_poisson_hmm(
            null_counts, num_states=num_states, num_restarts=num_restarts,
            num_em_iters=num_em_iters, seed=seed + it, verbose=verbose)
        state_probs, _ = decode_trial_state_probs(hmm, params, null_counts)
        late_state_idx = identify_late_state(state_probs, time_bins, late_window=late_window)
        onsets = find_late_state_onset(state_probs, late_state_idx, time_bins,
                                        prob_threshold=prob_threshold, min_time_s=min_time_s)

        found = onsets[~np.isnan(onsets)]
        max_prob_per_trial = state_probs[:, :, late_state_idx].max(axis=1)
        iqr = (np.percentile(found, 75) - np.percentile(found, 25)) if len(found) > 1 else np.nan

        result.per_iter.append(dict(
            iter=it, log_likelihood=ll, late_state_idx=late_state_idx,
            frac_trials_with_onset=len(found) / len(onsets),
            median_confidence=float(np.median(max_prob_per_trial)),
            onset_latency_iqr_s=float(iqr) if not np.isnan(iqr) else np.nan,
        ))
        print(f"[null iter {it + 1}/{n_null_iters}] LL={ll:.1f}, "
              f"late_state={late_state_idx}, "
              f"frac_with_onset={result.per_iter[-1]['frac_trials_with_onset']:.2f}, "
              f"median_confidence={result.per_iter[-1]['median_confidence']:.2f}")

    return result


if __name__ == "__main__":
    import pandas as pd

    rng = np.random.default_rng(0)
    n_units, n_trials = 15, 40
    fake_units = pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(n_units)],
        "spike_times": [np.sort(rng.uniform(0, 500, size=rng.integers(500, 2000)))
                         for _ in range(n_units)],
    })
    fake_trial_onsets = np.sort(rng.uniform(5, 490, size=n_trials))
    fake_bin_edges = np.arange(-0.2, 1.0, 0.02)

    result = run_null_control_test(
        fake_units, fake_trial_onsets, fake_bin_edges, num_states=2,
        num_restarts=3, num_em_iters=40, n_null_iters=3, seed=0,
    )
    result.summary()
