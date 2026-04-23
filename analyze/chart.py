import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import os

# ── Config ────────────────────────────────────────────────────────────────────
LOGS_DIR       = "../logs"
PLOTS_DIR      = "./plots"
WATCH_NEURONS  = [0, 10, 25, 50, 75]   # membrane/trace traces for these
WEIGHT_SNAPS   = [0, -1]               # first and last weight snapshot

os.makedirs(PLOTS_DIR, exist_ok=True)
plt.style.use('dark_background')

def load(name, required=False):
    path = f"{LOGS_DIR}/{name}"
    if not os.path.exists(path):
        if required: print(f"[WARN] Missing: {path}")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {name}: {len(df):,} rows")
    return df

spikes     = load("spikes.csv",     required=True)
membranes  = load("membranes.csv")
weights    = load("weights.csv")
traces     = load("traces.csv")
refractory = load("refractory.csv")
thresholds = load("thresholds.csv")
population = load("population.csv")

ACCENT  = '#00bbff'
ACCENT2 = '#ff3b00'
ACCENT3 = '#a78bfa'
ACCENT4 = '#34d399'

# ── 1. Raster Plot ────────────────────────────────────────────────────────────
if spikes is not None:
    fired = spikes[spikes['fired'] == 1]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.scatter(fired['t'], fired['neuron_id'], s=0.8, c=ACCENT, alpha=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Raster Plot — each dot is a spike")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b1_raster.png", dpi=150)
    plt.close()
    print("✓ 1_raster.png")

# ── 2. Population Summary (4-panel) ───────────────────────────────────────────
if population is not None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fig.suptitle("Population Summary", fontsize=14)

    axes[0,0].plot(population['t'], population['firing_rate'], color=ACCENT, lw=1)
    axes[0,0].fill_between(population['t'], population['firing_rate'], alpha=0.15, color=ACCENT)
    axes[0,0].set_title("Firing Rate")
    axes[0,0].set_ylabel("Spikes / neuron")

    axes[0,1].plot(population['t'], population['mean_membrane'], color=ACCENT4, lw=1)
    axes[0,1].set_title("Mean Membrane Potential")
    axes[0,1].set_ylabel("v")

    axes[1,0].plot(population['t'], population['mean_weight'], color=ACCENT3, lw=1)
    axes[1,0].set_title("Mean Synaptic Weight")
    axes[1,0].set_ylabel("w")

    axes[1,1].plot(population['t'], population['n_refractory'], color=ACCENT2, lw=1)
    axes[1,1].fill_between(population['t'], population['n_refractory'], alpha=0.15, color=ACCENT2)
    axes[1,1].set_title("Neurons in Refractory")
    axes[1,1].set_ylabel("count")

    for ax in axes.flat:
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b2_population_summary.png", dpi=150)
    plt.close()
    print("✓ 2_population_summary.png")

# ── 3. Membrane Traces ────────────────────────────────────────────────────────
if membranes is not None:
    fig, axes = plt.subplots(len(WATCH_NEURONS), 1,
                             figsize=(16, 2.5 * len(WATCH_NEURONS)), sharex=True)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(WATCH_NEURONS)))
    for ax, nid, col in zip(axes, WATCH_NEURONS, colors):
        df = membranes[membranes['neuron_id'] == nid]
        ax.plot(df['t'], df['v'], color=col, lw=0.8)
        ax.axhline(0, color='#444', lw=0.5, ls='--')
        ax.set_ylabel(f"N{nid}", fontsize=8)

        # overlay spike times
        if spikes is not None:
            sp = spikes[(spikes['neuron_id'] == nid) & (spikes['fired'] == 1)]
            for st in sp['t']:
                ax.axvline(st, color='white', alpha=0.3, lw=0.5)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Membrane Potential Traces (white lines = spikes)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b3_membrane_traces.png", dpi=150)
    plt.close()
    print("✓ 3_membrane_traces.png")

# ── 4. STDP Traces ────────────────────────────────────────────────────────────
if traces is not None:
    fig, axes = plt.subplots(len(WATCH_NEURONS), 1,
                             figsize=(16, 2.5 * len(WATCH_NEURONS)), sharex=True)
    for ax, nid in zip(axes, WATCH_NEURONS):
        df = traces[traces['neuron_id'] == nid]
        ax.plot(df['t'], df['pre_trace'],  color=ACCENT,  lw=0.8, label='pre')
        ax.plot(df['t'], df['post_trace'], color=ACCENT2, lw=0.8, label='post')
        ax.set_ylabel(f"N{nid}", fontsize=8)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("STDP Pre/Post Traces — should decay between spikes", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b4_stdp_traces.png", dpi=150)
    plt.close()
    print("✓ 4_stdp_traces.png")

# ── 5. Weight Distribution (start vs end) ─────────────────────────────────────
if weights is not None:
    timesteps = sorted(weights['t'].unique())
    snaps = [timesteps[i] for i in WEIGHT_SNAPS if abs(i) < len(timesteps)]

    fig, axes = plt.subplots(1, len(snaps), figsize=(7 * len(snaps), 5), sharey=True)
    if len(snaps) == 1: axes = [axes]

    for ax, t in zip(axes, snaps):
        w = weights[weights['t'] == t]['weight']
        ax.hist(w, bins=60, color='steelblue', edgecolor='none', alpha=0.85)
        ax.set_title(f"t = {t}  (n={len(w):,})")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")

    fig.suptitle("Weight Distribution — STDP pushes toward bimodal if working", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b5_weight_distribution.png", dpi=150)
    plt.close()
    print("✓ 5_weight_distribution.png")

# ── 6. Weight Matrix Heatmap ──────────────────────────────────────────────────
if weights is not None:
    timesteps = sorted(weights['t'].unique())
    snaps = [timesteps[i] for i in WEIGHT_SNAPS if abs(i) < len(timesteps)]

    fig, axes = plt.subplots(1, len(snaps), figsize=(8 * len(snaps), 6))
    if len(snaps) == 1: axes = [axes]

    for ax, t in zip(axes, snaps):
        w_snap = weights[weights['t'] == t].pivot_table(
            index='pre', columns='post', values='weight', aggfunc='mean'
        )
        sns.heatmap(w_snap, ax=ax, cmap='inferno', cbar=True,
                    xticklabels=False, yticklabels=False)
        ax.set_title(f"t = {t}")

    fig.suptitle("Weight Matrix — structure = learning, noise = no learning", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b6_weight_heatmap.png", dpi=150)
    plt.close()
    print("✓ 6_weight_heatmap.png")

# ── 7. Mean Weight Over Time ──────────────────────────────────────────────────
if weights is not None:
    w_over_time = weights.groupby('t')['weight'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(w_over_time['t'], w_over_time['mean'], color=ACCENT3, lw=1.2, label='mean')
    ax.fill_between(w_over_time['t'],
                    w_over_time['mean'] - w_over_time['std'],
                    w_over_time['mean'] + w_over_time['std'],
                    alpha=0.2, color=ACCENT3, label='±1 std')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Weight")
    ax.set_title("Mean Synaptic Weight ± Std — converging std = weights stabilising")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b7_weight_over_time.png", dpi=150)
    plt.close()
    print("✓ 7_weight_over_time.png")

# ── 8. Refractory Heatmap ─────────────────────────────────────────────────────
if refractory is not None:
    ref_pivot = refractory.pivot_table(
        index='neuron_id', columns='t', values='refractory', aggfunc='max'
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(ref_pivot, ax=ax, cmap='Reds', cbar=True,
                xticklabels=max(1, len(ref_pivot.columns)//20),
                yticklabels=max(1, len(ref_pivot.index)//20))
    ax.set_title("Refractory State — dark rows = neuron fires regularly")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron ID")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b8_refractory_heatmap.png", dpi=150)
    plt.close()
    print("✓ 8_refractory_heatmap.png")

# ── 9. Spikes per Neuron ──────────────────────────────────────────────────────
if spikes is not None:
    fired = spikes[spikes['fired'] == 1]
    spikes_per_neuron = fired.groupby('neuron_id').size().reindex(
        range(spikes['neuron_id'].max() + 1), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(spikes_per_neuron.index, spikes_per_neuron.values,
           color=ACCENT, width=1.0, alpha=0.8)
    ax.axhline(spikes_per_neuron.mean(), color='white', lw=1,
               ls='--', label=f"mean={spikes_per_neuron.mean():.1f}")
    ax.set_xlabel("Neuron ID")
    ax.set_ylabel("Total Spikes")
    ax.set_title("Spikes per Neuron — zeros=dead neurons, outliers=hyperactive")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b9_spikes_per_neuron.png", dpi=150)
    plt.close()
    print("✓ 9_spikes_per_neuron.png")

# ── 10. Threshold Drift ───────────────────────────────────────────────────────
if thresholds is not None:
    thresh_summary = thresholds.groupby('t')['threshold'].agg(['mean','min','max']).reset_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(thresh_summary['t'], thresh_summary['mean'], color=ACCENT4, lw=1.2, label='mean')
    ax.fill_between(thresh_summary['t'], thresh_summary['min'],
                    thresh_summary['max'], alpha=0.15, color=ACCENT4, label='min/max')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold Drift — rising=homeostatic adaptation, flat=no adaptation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/b10_threshold_drift.png", dpi=150)
    plt.close()
    print("✓ 10_threshold_drift.png")

print(f"\nAll plots saved to {PLOTS_DIR}/")
