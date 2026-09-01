"""
HMM Helper Module for Auditory Cortex Population Analysis using Dynamax.
Supports PoissonHMM fitting, multi-restart model selection, posterior state decoding,
state characterization, and post-HMM spike train realignment (PSTH vs PTTH).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model import PoissonHMM


def initialize_and_fit_poisson_hmm(
    counts,
    num_states,
    bin_width=0.01,
    n_restarts=5,
    num_iters=60,
    diagonal_prior=0.95,
    seed=42
):
    """
    Fits a PoissonHMM on binned spike counts across multiple trials with multi-restart.

    Parameters:
    -----------
    counts : np.ndarray
        Shape (num_trials, num_timesteps, num_units) of integer spike counts.
    num_states : int
        Number of hidden states (K).
    bin_width : float
        Bin width in seconds (e.g., 0.01 for 10 ms).
    n_restarts : int
        Number of random initializations.
    num_iters : int
        Max EM iterations.
    diagonal_prior : float
        Initial diagonal persistence probability D for transition matrix.
    seed : int
        Base random seed.

    Returns:
    --------
    best_model : PoissonHMM
    best_params : HMMParameterSet
    best_ll : float
    lls_trace : list
    """
    jnp_counts = jnp.array(counts, dtype=jnp.int32)
    num_trials, num_timesteps, emission_dim = counts.shape
    mean_rate = np.maximum(counts.mean(axis=(0, 1)), 1e-4) # baseline empirical counts per bin

    best_ll = -np.inf
    best_params = None
    best_lls_trace = None
    best_hmm = None

    for r in range(n_restarts):
        key = jr.PRNGKey(seed + r * 137)
        hmm = PoissonHMM(num_states, emission_dim)

        # Persistent transition matrix prior
        T_init = np.full((num_states, num_states), (1.0 - diagonal_prior) / max(1, num_states - 1))
        np.fill_diagonal(T_init, diagonal_prior)

        # Perturbed empirical emission rates
        np.random.seed(seed + r * 137)
        rates_init = np.outer(np.ones(num_states), mean_rate) * np.random.uniform(0.6, 1.4, size=(num_states, emission_dim))
        rates_init = np.maximum(rates_init, 1e-4)

        params, props = hmm.initialize(
            key=key,
            initial_probs=jnp.full(num_states, 1.0 / num_states),
            transition_matrix=jnp.array(T_init, dtype=jnp.float32),
            emission_rates=jnp.array(rates_init, dtype=jnp.float32)
        )

        params, lls = hmm.fit_em(params, props, jnp_counts, num_iters=num_iters, verbose=False)
        final_ll = float(lls[-1])

        if final_ll > best_ll:
            best_ll = final_ll
            best_params = params
            best_lls_trace = lls
            best_hmm = hmm

    return best_hmm, best_params, best_ll, best_lls_trace


def compute_posterior_state_probabilities(hmm, params, counts):
    """
    Computes smoothed posterior state probabilities gamma(trial, t, s) = P(s_t = s | obs).

    Returns:
    --------
    state_probs : np.ndarray
        Shape (num_trials, num_timesteps, num_states).
    """
    import jax
    jnp_counts = jnp.array(counts, dtype=jnp.int32)
    
    def _smooth_single_trial(c):
        return hmm.smoother(params, c).smoothed_probs

    vmapped_smoother = jax.vmap(_smooth_single_trial)
    smoothed_probs = vmapped_smoother(jnp_counts)
    return np.array(smoothed_probs)


def reorder_states_temporally(state_probs, params, bin_centers):
    """
    Reorders hidden states based on their center-of-mass or peak probability across time
    so that State 1 = earliest active state (e.g. baseline), State K = latest active state.

    Returns:
    --------
    reordered_probs : np.ndarray, shape (num_trials, num_timesteps, num_states)
    reordered_transitions : np.ndarray, shape (num_states, num_states)
    reordered_rates_hz : np.ndarray, shape (num_states, num_units)
    order : np.ndarray of indices
    """
    num_trials, num_timesteps, num_states = state_probs.shape
    mean_prob_over_time = state_probs.mean(axis=0) # (num_timesteps, num_states)
    
    # Calculate time-weighted center of mass for each state
    center_of_mass = np.array([np.sum(bin_centers * mean_prob_over_time[:, s]) / np.sum(mean_prob_over_time[:, s])
                               for s in range(num_states)])
    order = np.argsort(center_of_mass)

    reordered_probs = state_probs[:, :, order]
    T = np.array(params.transitions.transition_matrix)
    reordered_transitions = T[order, :][:, order]

    # Convert lambda counts per 10ms bin to Hz (rate = lambda / bin_width)
    emission_rates = np.array(params.emissions.rates) # (num_states, num_units)
    reordered_rates_hz = emission_rates[order, :] / 0.01

    return reordered_probs, reordered_transitions, reordered_rates_hz, order


def detect_state_onsets(state_probs, bin_centers, target_state, prob_thresh=0.5, min_onset_s=0.1):
    """
    Finds the first time bin where P(target_state) > prob_thresh after sound onset (t >= 0).
    Filters out trials where state onset < min_onset_s.

    Returns:
    --------
    onset_table : pd.DataFrame
        Columns: trial_idx, onset_time_s, onset_bin, is_valid
    """
    num_trials, num_timesteps, _ = state_probs.shape
    sound_onset_bin = np.searchsorted(bin_centers, 0.0)

    records = []
    for trial_idx in range(num_trials):
        probs_post_stim = state_probs[trial_idx, sound_onset_bin:, target_state]
        times_post_stim = bin_centers[sound_onset_bin:]
        
        crossings = np.where(probs_post_stim >= prob_thresh)[0]
        if len(crossings) > 0:
            first_crossing_idx = crossings[0]
            onset_time = float(times_post_stim[first_crossing_idx])
            onset_bin = sound_onset_bin + first_crossing_idx
            is_valid = (onset_time >= min_onset_s)
        else:
            onset_time = np.nan
            onset_bin = -1
            is_valid = False

        records.append({
            'trial_idx': trial_idx,
            'onset_time_s': onset_time,
            'onset_bin': onset_bin,
            'is_valid': is_valid
        })

    return pd.DataFrame(records)


def realign_spike_times(spike_times_df, onset_df, trial_ids):
    """
    Realigns long-format spike times to detected state onsets (PTTH alignment).

    Parameters:
    -----------
    spike_times_df : pd.DataFrame
        Columns: unit_id, trial_id, spike_time (relative to sound onset).
    onset_df : pd.DataFrame
        Columns: trial_idx, onset_time_s, is_valid.
    trial_ids : np.ndarray
        Array mapping trial_idx to original trial_id.

    Returns:
    --------
    realigned_df : pd.DataFrame
        Columns: unit_id, trial_id, spike_time_stim, spike_time_state, is_valid
    """
    onset_lookup = {}
    for _, row in onset_df.iterrows():
        t_id = trial_ids[int(row['trial_idx'])]
        onset_lookup[t_id] = (row['onset_time_s'], row['is_valid'])

    df = spike_times_df.copy()
    onsets = [onset_lookup.get(t, (np.nan, False))[0] for t in df['trial_id']]
    valids = [onset_lookup.get(t, (np.nan, False))[1] for t in df['trial_id']]

    df['state_onset_time'] = onsets
    df['is_valid_trial'] = valids
    df['spike_time_state_aligned'] = df['spike_time'] - df['state_onset_time']

    return df


def compute_all_state_onsets(state_probs, bin_centers, trial_ids, prob_thresh=0.5):
    """
    Computes onset latency for every hidden state across all trials.

    Parameters:
    -----------
    state_probs : np.ndarray, shape (num_trials, num_timesteps, num_states)
    bin_centers : np.ndarray, shape (num_timesteps,)
    trial_ids : np.ndarray, shape (num_trials,)
    prob_thresh : float, probability threshold to define onset (default: 0.5)

    Returns:
    --------
    onsets_dict : dict mapping state_idx -> np.ndarray of onset times in seconds (or NaN if never exceeded)
    summary_df : pd.DataFrame with summary statistics (mean, median, std, IQR) per state
    """
    num_trials, num_timesteps, num_states = state_probs.shape
    onsets_dict = {}
    records = []

    for s in range(num_states):
        onset_times = []
        for t_idx in range(num_trials):
            p_trace = state_probs[t_idx, :, s]
            crossings = np.where(p_trace >= prob_thresh)[0]
            if len(crossings) > 0:
                onset_times.append(bin_centers[crossings[0]])
            else:
                onset_times.append(np.nan)

        arr = np.array(onset_times)
        onsets_dict[s] = arr
        valid = arr[~np.isnan(arr)]

        records.append({
            'State': f'State {s+1}',
            'Occurred_in_trials': f"{len(valid)} / {num_trials} ({len(valid)/num_trials*100:.1f}%)",
            'Mean_Onset_ms': np.mean(valid) * 1000 if len(valid) > 0 else np.nan,
            'Median_Onset_ms': np.median(valid) * 1000 if len(valid) > 0 else np.nan,
            'Std_ms': np.std(valid) * 1000 if len(valid) > 0 else np.nan,
            'IQR_ms': (np.percentile(valid, 75) - np.percentile(valid, 25)) * 1000 if len(valid) > 0 else np.nan
        })

    summary_df = pd.DataFrame(records)
    return onsets_dict, summary_df


def plot_all_state_onset_distributions(onsets_s1_dict, onsets_s2_dict, num_states, save_path=None):
    """
    Plots Probability Density (KDE), Cumulative Probability (ECDF), and Boxplots
    of onset latencies for all hidden states across both sound conditions.
    """
    state_colors = sns.color_palette("tab10", num_states)
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.22)

    for col, (onsets_per_state, label) in enumerate([
        (onsets_s1_dict, 'Sound 1'),
        (onsets_s2_dict, 'Sound 2')
    ]):
        # 1. Probability Density (KDE)
        ax_kde = axes[0, col]
        for s in range(num_states):
            valid = onsets_per_state[s][~np.isnan(onsets_per_state[s])] * 1000 # to ms
            if len(valid) > 1:
                sns.kdeplot(valid, ax=ax_kde, color=state_colors[s], lw=2.5,
                            label=f'State {s+1} (Med: {np.median(valid):.0f} ms)', fill=True, alpha=0.15)
                ax_kde.scatter(valid, np.zeros_like(valid) - 0.0002 * (s+1), color=state_colors[s], alpha=0.5, s=15, marker='|')

        ax_kde.axvline(0, color='black', ls='--', lw=1.5, alpha=0.7, label='Sound Onset')
        ax_kde.set_title(f'{label}: Onset Probability Density Function (K={num_states})', fontsize=11, fontweight='bold')
        ax_kde.set_xlabel('Onset Latency re: Sound Onset (ms)', fontsize=9)
        ax_kde.set_ylabel('Probability Density', fontsize=9)
        ax_kde.set_xlim(-550, 1550)
        ax_kde.legend(loc='upper right', fontsize=8, frameon=True)

        # 2. Empirical Cumulative Distribution Function (ECDF)
        ax_ecdf = axes[1, col]
        for s in range(num_states):
            valid = np.sort(onsets_per_state[s][~np.isnan(onsets_per_state[s])] * 1000)
            if len(valid) > 0:
                ecdf_y = np.arange(1, len(valid) + 1) / len(valid)
                ax_ecdf.step(valid, ecdf_y, where='post', color=state_colors[s], lw=2.2, label=f'State {s+1}')

        ax_ecdf.axvline(0, color='black', ls='--', lw=1.5, alpha=0.7)
        ax_ecdf.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.7)
        ax_ecdf.set_title(f'{label}: Cumulative Onset Probability (ECDF)', fontsize=11, fontweight='bold')
        ax_ecdf.set_xlabel('Onset Latency re: Sound Onset (ms)', fontsize=9)
        ax_ecdf.set_ylabel('Cumulative Probability', fontsize=9)
        ax_ecdf.set_xlim(-550, 1550)
        ax_ecdf.set_ylim(-0.02, 1.02)
        ax_ecdf.legend(loc='lower right', fontsize=8, frameon=True)

        # 3. Boxplot & Trial Points
        ax_box = axes[2, col]
        box_data = []
        for s in range(num_states):
            valid = onsets_per_state[s][~np.isnan(onsets_per_state[s])] * 1000
            for v in valid:
                box_data.append({'State': f'State {s+1}', 'Onset_ms': v})
        df_box = pd.DataFrame(box_data)
        sns.boxplot(x='State', y='Onset_ms', data=df_box, ax=ax_box, width=0.45, color='white',
                    boxprops=dict(alpha=0.8, edgecolor='black'))
        for s in range(num_states):
            s_pts = [d['Onset_ms'] for d in box_data if d['State'] == f'State {s+1}']
            if len(s_pts) > 0:
                ax_box.scatter(np.random.normal(s, 0.08, size=len(s_pts)), s_pts,
                               color=state_colors[s], alpha=0.5, s=16)

        ax_box.axhline(0, color='black', ls='--', lw=1.5, alpha=0.7)
        ax_box.set_title(f'{label}: State Onset Latencies & Trial Variance', fontsize=11, fontweight='bold')
        ax_box.set_xlabel('Hidden State', fontsize=9)
        ax_box.set_ylabel('Onset Time re: Sound Onset (ms)', fontsize=9)
        ax_box.set_ylim(-550, 1550)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig, axes


def plot_trial_posterior_heatmaps(probs_s1, probs_s2, trial_ids_s1, trial_ids_s2, bin_centers, num_states, save_path=None):
    """
    Generates static trial-by-trial posterior state heatmaps and dominant state rasters
    for both Sound 1 and Sound 2.
    """
    import matplotlib.colors as mcolors
    times_ms = bin_centers * 1000
    palette = sns.color_palette("tab10", num_states)
    cmap_discrete = mcolors.ListedColormap(palette)
    bounds = np.arange(num_states + 1) - 0.5
    norm_discrete = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

    dom_s1 = np.argmax(probs_s1, axis=-1)
    dom_s2 = np.argmax(probs_s2, axis=-1)

    s3_onsets_s1 = [np.where(probs_s1[t, :, min(2, num_states-1)] >= 0.5)[0][0] if len(np.where(probs_s1[t, :, min(2, num_states-1)] >= 0.5)[0]) > 0 else 9999 for t in range(len(trial_ids_s1))]
    order_s1 = np.argsort(s3_onsets_s1)

    s3_onsets_s2 = [np.where(probs_s2[t, :, min(2, num_states-1)] >= 0.5)[0][0] if len(np.where(probs_s2[t, :, min(2, num_states-1)] >= 0.5)[0]) > 0 else 9999 for t in range(len(trial_ids_s2))]
    order_s2 = np.argsort(s3_onsets_s2)

    fig = plt.figure(figsize=(18, 14), dpi=150)
    gs = fig.add_gridspec(num_states + 1, 2, height_ratios=[2.2] + [1.0] * num_states, hspace=0.35, wspace=0.18)

    for col, (probs, dom, order, t_ids, label, c_name) in enumerate([
        (probs_s1, dom_s1, order_s1, trial_ids_s1, 'Sound 1', '#1f77b4'),
        (probs_s2, dom_s2, order_s2, trial_ids_s2, 'Sound 2', '#ff7f0e')
    ]):
        ax_top = fig.add_subplot(gs[0, col])
        im_dom = ax_top.imshow(
            dom[order], aspect='auto', cmap=cmap_discrete, norm=norm_discrete,
            extent=[times_ms[0], times_ms[-1], len(t_ids), 0], interpolation='nearest'
        )
        ax_top.axvline(0, color='white', ls='--', lw=1.8, label='Sound Onset')
        ax_top.set_title(f'{label}: Dominant State Sequence ({len(t_ids)} Trials, K={num_states})',
                         fontsize=12, fontweight='bold', color=c_name)
        ax_top.set_ylabel('Trials (Sorted by Evoked State Onset)', fontsize=9)
        if col == 1:
            cbar = fig.colorbar(im_dom, ax=ax_top, orientation='vertical', pad=0.02, ticks=range(num_states))
            cbar.ax.set_yticklabels([f'State {s+1}' for s in range(num_states)], fontsize=8)

        for s in range(num_states):
            ax_s = fig.add_subplot(gs[s + 1, col], sharex=ax_top)
            im_s = ax_s.imshow(
                probs[order, :, s], aspect='auto', cmap='magma', vmin=0, vmax=1.0,
                extent=[times_ms[0], times_ms[-1], len(t_ids), 0], interpolation='nearest'
            )
            ax_s.axvline(0, color='cyan', ls='--', lw=1.2, alpha=0.8)
            ax_s.set_ylabel(f'S{s+1} Trials', fontsize=8)
            ax_s.set_title(f'P(State {s+1}) Posterior Probability', fontsize=9, loc='left', pad=2)
            if col == 1 and s == 0:
                cbar_s = fig.colorbar(im_s, ax=ax_s, orientation='vertical', pad=0.02, ticks=[0, 0.5, 1.0])
                cbar_s.ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=7)
            if s == num_states - 1:
                ax_s.set_xlabel('Time re: Sound Onset (ms)', fontsize=10, fontweight='bold')

    plt.suptitle(f'Posterior State Probabilities Across All Trials (K = {num_states} PoissonHMM)', fontsize=14, fontweight='bold', y=0.995)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_interactive_posterior_dashboard(probs_s1, probs_s2, trial_ids_s1, trial_ids_s2, bin_centers, df_trials_s1=None, df_trials_s2=None, num_states=5, save_html_path=None):
    """
    Generates an interactive Plotly HTML dashboard showing full trial rasters with hover tooltips
    and optional behavioral response markers.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    times_ms = bin_centers * 1000

    dom_s1 = np.argmax(probs_s1, axis=-1) + 1
    dom_s2 = np.argmax(probs_s2, axis=-1) + 1

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Sound 1: Trial-by-Trial Dominant State Raster ({len(trial_ids_s1)} Trials)",
            f"Sound 2: Trial-by-Trial Dominant State Raster ({len(trial_ids_s2)} Trials)"
        ],
        vertical_spacing=0.10,
        shared_xaxes=True
    )

    state_colorscale = [
        [0.0, '#1f77b4'],
        [0.25, '#ff7f0e'],
        [0.5, '#2ca02c'],
        [0.75, '#d62728'],
        [1.0, '#9467bd']
    ] if num_states == 5 else [
        [0.0, '#1f77b4'],
        [0.33, '#ff7f0e'],
        [0.66, '#2ca02c'],
        [1.0, '#d62728']
    ]

    custom_hover_s1 = []
    for t_idx, t_id in enumerate(trial_ids_s1):
        row = []
        t_meta = df_trials_s1[df_trials_s1['id'] == t_id].iloc[0] if df_trials_s1 is not None else None
        rt_str = f"{t_meta['rt_ms']:.1f} ms" if (t_meta is not None and pd.notna(t_meta.get('rt_ms'))) else "No Response"
        outcome = ("Hit" if t_meta['is_hit'] else ("False Alarm" if t_meta['is_false_alarm'] else ("Miss" if t_meta['is_miss'] else "Correct Reject"))) if t_meta is not None else "N/A"
        mod = t_meta['rewarded_modality'].upper() if t_meta is not None else "N/A"

        for b_idx, t_ms in enumerate(times_ms):
            s_dom = dom_s1[t_idx, b_idx]
            p_val = probs_s1[t_idx, b_idx, s_dom - 1]
            row.append(
                f"<b>Trial ID: {t_id}</b> (Trial {t_idx+1})<br>"
                f"Time: {t_ms:.1f} ms<br>"
                f"Dominant State: <b>State {s_dom}</b> (P={p_val:.3f})<br>"
                f"Outcome: {outcome}<br>"
                f"Behavioral RT: {rt_str}<br>"
                f"Context: {mod}"
            )
        custom_hover_s1.append(row)

    trace_s1 = go.Heatmap(
        z=dom_s1,
        x=times_ms,
        y=[f"T{tid}" for tid in trial_ids_s1],
        text=custom_hover_s1,
        hoverinfo="text",
        colorscale=state_colorscale,
        zmin=1, zmax=num_states,
        colorbar=dict(
            title=dict(text="Dominant State", side="top"),
            tickvals=list(range(1, num_states + 1)),
            ticktext=[f"State {s}" for s in range(1, num_states + 1)],
            len=0.75,
            x=1.02
        ),
        name="Sound 1 States"
    )

    custom_hover_s2 = []
    for t_idx, t_id in enumerate(trial_ids_s2):
        row = []
        t_meta = df_trials_s2[df_trials_s2['id'] == t_id].iloc[0] if df_trials_s2 is not None else None
        rt_str = f"{t_meta['rt_ms']:.1f} ms" if (t_meta is not None and pd.notna(t_meta.get('rt_ms'))) else "No Response"
        outcome = ("Hit" if t_meta['is_hit'] else ("False Alarm" if t_meta['is_false_alarm'] else ("Miss" if t_meta['is_miss'] else "Correct Reject"))) if t_meta is not None else "N/A"
        mod = t_meta['rewarded_modality'].upper() if t_meta is not None else "N/A"

        for b_idx, t_ms in enumerate(times_ms):
            s_dom = dom_s2[t_idx, b_idx]
            p_val = probs_s2[t_idx, b_idx, s_dom - 1]
            row.append(
                f"<b>Trial ID: {t_id}</b> (Trial {t_idx+1})<br>"
                f"Time: {t_ms:.1f} ms<br>"
                f"Dominant State: <b>State {s_dom}</b> (P={p_val:.3f})<br>"
                f"Outcome: {outcome}<br>"
                f"Behavioral RT: {rt_str}<br>"
                f"Context: {mod}"
            )
        custom_hover_s2.append(row)

    trace_s2 = go.Heatmap(
        z=dom_s2,
        x=times_ms,
        y=[f"T{tid}" for tid in trial_ids_s2],
        text=custom_hover_s2,
        hoverinfo="text",
        colorscale=state_colorscale,
        zmin=1, zmax=num_states,
        showscale=False,
        name="Sound 2 States"
    )

    fig.add_trace(trace_s1, row=1, col=1)
    fig.add_trace(trace_s2, row=2, col=1)

    # Add behavioral markers if trial dataframes provided
    if df_trials_s1 is not None:
        s1_hits_x, s1_hits_y, s1_hits_h = [], [], []
        s1_fa_x, s1_fa_y, s1_fa_h = [], [], []
        for t_idx, t_id in enumerate(trial_ids_s1):
            t_meta = df_trials_s1[df_trials_s1['id'] == t_id].iloc[0]
            if pd.notna(t_meta.get('rt_ms')):
                h_text = f"<b>Behavioral Response</b><br>Trial ID: {t_id}<br>RT: {t_meta['rt_ms']:.1f} ms<br>Outcome: {'Hit' if t_meta['is_hit'] else 'False Alarm'}"
                if t_meta['is_hit']:
                    s1_hits_x.append(t_meta['rt_ms'])
                    s1_hits_y.append(f"T{t_id}")
                    s1_hits_h.append(h_text)
                else:
                    s1_fa_x.append(t_meta['rt_ms'])
                    s1_fa_y.append(f"T{t_id}")
                    s1_fa_h.append(h_text)

        if len(s1_hits_x) > 0:
            fig.add_trace(go.Scatter(
                x=s1_hits_x, y=s1_hits_y, mode='markers',
                marker=dict(symbol='diamond', size=9, color='#00ffff', line=dict(color='black', width=1.5)),
                name='Hit Response (Lick)', hovertext=s1_hits_h, hoverinfo='text'
            ), row=1, col=1)
        if len(s1_fa_x) > 0:
            fig.add_trace(go.Scatter(
                x=s1_fa_x, y=s1_fa_y, mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#ff00ff', line=dict(color='black', width=1.5)),
                name='False Alarm Response', hovertext=s1_fa_h, hoverinfo='text'
            ), row=1, col=1)

    if df_trials_s2 is not None:
        s2_hits_x, s2_hits_y, s2_hits_h = [], [], []
        s2_fa_x, s2_fa_y, s2_fa_h = [], [], []
        for t_idx, t_id in enumerate(trial_ids_s2):
            t_meta = df_trials_s2[df_trials_s2['id'] == t_id].iloc[0]
            if pd.notna(t_meta.get('rt_ms')):
                h_text = f"<b>Behavioral Response</b><br>Trial ID: {t_id}<br>RT: {t_meta['rt_ms']:.1f} ms<br>Outcome: {'Hit' if t_meta['is_hit'] else 'False Alarm'}"
                if t_meta['is_hit']:
                    s2_hits_x.append(t_meta['rt_ms'])
                    s2_hits_y.append(f"T{t_id}")
                    s2_hits_h.append(h_text)
                else:
                    s2_fa_x.append(t_meta['rt_ms'])
                    s2_fa_y.append(f"T{t_id}")
                    s2_fa_h.append(h_text)

        if len(s2_hits_x) > 0:
            fig.add_trace(go.Scatter(
                x=s2_hits_x, y=s2_hits_y, mode='markers',
                marker=dict(symbol='diamond', size=9, color='#00ffff', line=dict(color='black', width=1.5)),
                showlegend=False, hovertext=s2_hits_h, hoverinfo='text'
            ), row=2, col=1)
        if len(s2_fa_x) > 0:
            fig.add_trace(go.Scatter(
                x=s2_fa_x, y=s2_fa_y, mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#ff00ff', line=dict(color='black', width=1.5)),
                showlegend=False, hovertext=s2_fa_h, hoverinfo='text'
            ), row=2, col=1)

    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="white", row="all", col="all")

    fig.update_layout(
        title=dict(
            text=f"Interactive Auditory Cortex Latent States with Behavioral Response Overlay (K={num_states})",
            font=dict(size=18, family="Arial")
        ),
        height=1000,
        width=1200,
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Time relative to Sound Onset (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Trials (Sound 1)", row=1, col=1)
    fig.update_yaxes(title_text="Trials (Sound 2)", row=2, col=1)

    if save_html_path:
        fig.write_html(save_html_path)
    return fig


