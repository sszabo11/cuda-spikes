#plt.scatter(spike_df['t'], spike_df['neuron_idx'], s=1)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import os

# ── Config ────────────────────────────────────────────────────────────────────
SPIKE_CSV      = "../logs/log.csv"       # t, neuron_idx, fired
MEMBRANE_CSV   = "membranes.csv"    # t, neuron_idx, v
WEIGHT_CSV     = "weights.csv"      # t, pre, post, weight
FIRING_CSV     = "firing_rate.csv"  # t, rate
WATCH_NEURONS  = [0, 50, 100, 200, 500]  # neurons to trace individually
WEIGHT_SNAPS   = [0, -1]            # timestep indices to compare (first, last)

os.makedirs("plots", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
def load(path, required=True):
    if not os.path.exists(path):
        if required:
            print(f"[WARN] Missing: {path}")
        return None
    return pd.read_csv(path)

spikes   = load(SPIKE_CSV)
membrane = load(MEMBRANE_CSV,  required=False)
weights  = load(WEIGHT_CSV,    required=False)
firing   = load(FIRING_CSV,    required=False)

# ── 1. Raster Plot ────────────────────────────────────────────────────────────
if spikes is not None:
    fired = spikes[spikes['fired'] == 1]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(fired['t'], fired['neuron_idx'], s=1, c='white', alpha=0.7)
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_xlabel("Timestep", color='grey')
    ax.set_ylabel("Neuron ID", color='grey')
    ax.set_title("Raster Plot", color='white', fontsize=14)
    ax.tick_params(colors='grey')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    plt.tight_layout()
    plt.savefig("plots/1_raster.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✓ Raster plot saved")

# ── 2. Membrane Potential Traces ──────────────────────────────────────────────
if membrane is not None:
    fig, axes = plt.subplots(len(WATCH_NEURONS), 1,
                             figsize=(14, 2.5 * len(WATCH_NEURONS)),
                             sharex=True)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(WATCH_NEURONS)))
    for ax, nid, col in zip(axes, WATCH_NEURONS, colors):
        df = membrane[membrane['neuron_idx'] == nid]
        ax.plot(df['t'], df['v'], color=col, linewidth=0.8)
        ax.set_ylabel(f"N{nid}", fontsize=8)
        ax.axhline(0, color='#444', linewidth=0.5, linestyle='--')
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='grey', labelsize=7)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Membrane Potential Traces", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/2_membrane_traces.png", dpi=150)
    plt.close()
    print("✓ Membrane traces saved")

# ── 3. Weight Distribution (start vs end) ─────────────────────────────────────
if weights is not None:
    timesteps = sorted(weights['t'].unique())
    snaps = [timesteps[i] for i in WEIGHT_SNAPS if abs(i) < len(timesteps)]
    fig, axes = plt.subplots(1, len(snaps), figsize=(6 * len(snaps), 5),
                             sharey=True)
    if len(snaps) == 1:
        axes = [axes]
    for ax, t in zip(axes, snaps):
        w = weights[weights['t'] == t]['weight']
        ax.hist(w, bins=60, color='steelblue', edgecolor='none', alpha=0.85)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
    fig.suptitle("Weight Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/3_weight_distribution.png", dpi=150)
    plt.close()
    print("✓ Weight distribution saved")

# ── 4. Weight Matrix Heatmap ──────────────────────────────────────────────────
if weights is not None:
    timesteps = sorted(weights['t'].unique())
    snaps = [timesteps[i] for i in WEIGHT_SNAPS if abs(i) < len(timesteps)]
    fig, axes = plt.subplots(1, len(snaps), figsize=(7 * len(snaps), 6))
    if len(snaps) == 1:
        axes = [axes]
    for ax, t in zip(axes, snaps):
        w_snap = weights[weights['t'] == t].pivot_table(
            index='pre', columns='post', values='weight', aggfunc='mean'
        )
        sns.heatmap(w_snap, ax=ax, cmap='inferno', cbar=True,
                    xticklabels=False, yticklabels=False)
        ax.set_title(f"Weight Matrix t={t}")
    fig.suptitle("Weight Matrix Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/4_weight_heatmap.png", dpi=150)
    plt.close()
    print("✓ Weight heatmap saved")

# ── 5. Population Firing Rate ─────────────────────────────────────────────────
if firing is not None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(firing['t'], firing['rate'], color='tomato', linewidth=1)
    ax.fill_between(firing['t'], firing['rate'], alpha=0.15, color='tomato')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Firing Rate")
    ax.set_title("Population Firing Rate Over Time")
    plt.tight_layout()
    plt.savefig("plots/5_firing_rate.png", dpi=150)
    plt.close()
    print("✓ Firing rate saved")
elif spikes is not None:
    # derive from spike data if no dedicated CSV
    rate = spikes.groupby('t')['fired'].mean().reset_index()
    rate.columns = ['t', 'rate']
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rate['t'], rate['rate'], color='tomato', linewidth=1)
    ax.fill_between(rate['t'], rate['rate'], alpha=0.15, color='tomato')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Firing Rate")
    ax.set_title("Population Firing Rate (derived from spikes)")
    plt.tight_layout()
    plt.savefig("plots/5_firing_rate.png", dpi=150)
    plt.close()
    print("✓ Firing rate (derived) saved")

# ── 6. Weight Delta Over Time ─────────────────────────────────────────────────
if weights is not None and 'delta_w' in weights.columns:
    dw = weights.groupby('t')['delta_w'].apply(lambda x: np.mean(np.abs(x)))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dw.index, dw.values, color='mediumpurple', linewidth=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean |Δw|")
    ax.set_title("Weight Change Over Time (STDP convergence)")
    plt.tight_layout()
    plt.savefig("plots/6_weight_delta.png", dpi=150)
    plt.close()
    print("✓ Weight delta saved")

# ── 7. Summary Dashboard ──────────────────────────────────────────────────────
if spikes is not None:
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#111')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # raster
    ax1 = fig.add_subplot(gs[0, :])
    fired = spikes[spikes['fired'] == 1]
    ax1.scatter(fired['t'], fired['neuron_idx'], s=0.5, c='cyan', alpha=0.5)
    ax1.set_facecolor('#0a0a0a')
    ax1.set_title("Raster", color='white')
    ax1.tick_params(colors='grey')

    # firing rate
    rate = spikes.groupby('t')['fired'].mean()
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rate.index, rate.values, color='tomato', linewidth=1)
    ax2.fill_between(rate.index, rate.values, alpha=0.2, color='tomato')
    ax2.set_facecolor('#0a0a0a')
    ax2.set_title("Firing Rate", color='white')
    ax2.tick_params(colors='grey')

    # spikes per neuron histogram
    ax3 = fig.add_subplot(gs[1, 1])
    spikes_per_neuron = fired.groupby('neuron_idx').size()
    ax3.bar(spikes_per_neuron.index, spikes_per_neuron.values,
            color='steelblue', width=1.0)
    ax3.set_facecolor('#0a0a0a')
    ax3.set_title("Spikes per Neuron", color='white')
    ax3.set_xlabel("Neuron ID", color='grey')
    ax3.tick_params(colors='grey')

    plt.suptitle("SNN Summary Dashboard", color='white', fontsize=15)
    plt.savefig("plots/7_dashboard.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✓ Dashboard saved")

print("\nAll plots saved to ./plots/")
