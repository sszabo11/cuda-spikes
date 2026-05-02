"""
analyze.py — SNN digit response analysis
Run from the analyze/ directory:  python analyze.py

Reads:
  ../logs/digit_responses.csv   (img_idx, label, n0..n783)
  ../logs/population.csv        (t, firing_rate, mean_membrane, ...)

Produces (saved to plots/):
  01_receptive_fields.png       mean spike vector per digit, shown as 28x28 image
  02_similarity_matrix.png      10x10 cosine similarity between digit responses
  03_spike_rate_bars.png        per-neuron mean firing rate, grouped by digit
  04_pca.png                    PCA of response vectors, coloured by digit
  05_tsne.png                   t-SNE of response vectors, coloured by digit
  06_training_dynamics.png      firing rate / membrane / weight over time
  07_selectivity_map.png        per-neuron preferred digit shown as 28x28 image
  08_within_digit_consistency.png  intra-class cosine similarity distributions
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR     = os.path.join(SCRIPT_DIR, "..", "logs")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
RESPONSES    = os.path.join(LOGS_DIR, "digit_responses.csv")
POPULATION   = os.path.join(LOGS_DIR, "population.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# ── helpers ────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_responses():
    if not os.path.exists(RESPONSES):
        print(f"ERROR: {RESPONSES} not found. Run the network first.")
        sys.exit(1)

    data = np.genfromtxt(RESPONSES, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")

    labels      = data["label"].astype(int)
    img_indices = data["img_idx"].astype(int)

    # spike count columns are named n0, n1, ..., n783
    n_neurons = sum(1 for name in data.dtype.names if name.startswith("n"))
    vectors   = np.column_stack([data[f"n{i}"].astype(float)
                                 for i in range(n_neurons)])

    return img_indices, labels, vectors, n_neurons


def mean_vectors(labels, vectors):
    """Return dict digit → mean spike vector."""
    mv = {}
    for d in range(10):
        mask = labels == d
        if mask.sum() > 0:
            mv[d] = vectors[mask].mean(axis=0)
        else:
            mv[d] = np.zeros(vectors.shape[1])
    return mv


# ── plot 1 — receptive fields ──────────────────────────────────────────────────

def plot_receptive_fields(labels, vectors, n_neurons):
    print("Plot 1: receptive fields")
    mv = mean_vectors(labels, vectors)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Mean spike response per digit  (28×28 neuron map)", fontsize=14)

    for d, ax in enumerate(axes.flat):
        img  = mv[d].reshape(28, 28)
        mask = labels == d
        rate = vectors[mask].sum(axis=1).mean() if mask.sum() > 0 else 0
        im   = ax.imshow(img, cmap="hot", interpolation="nearest")
        ax.set_title(f"Digit {d}  (mean spikes/img: {rate:.1f})", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save(fig, "z_01_receptive_fields.png")


# ── plot 2 — cosine similarity matrix ─────────────────────────────────────────

def plot_similarity_matrix(labels, vectors):
    print("Plot 2: similarity matrix")
    mv   = mean_vectors(labels, vectors)
    sim  = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            sim[i, j] = cosine_sim(mv[i], mv[j])

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Pairwise cosine similarity of mean digit responses", fontsize=13)
    cax = ax.imshow(sim, vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
    fig.colorbar(cax, ax=ax, label="Cosine similarity")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_yticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Digit")

    save(fig, "z_02_similarity_matrix.png")


# ── plot 3 — mean spike rate bar chart per digit ───────────────────────────────

def plot_spike_rate_bars(labels, vectors, n_neurons):
    print("Plot 3: spike rate bars")
    mv = mean_vectors(labels, vectors)

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharey=True)
    fig.suptitle("Mean spike count per neuron, grouped by digit", fontsize=13)

    neuron_ids = np.arange(n_neurons)
    for d, ax in enumerate(axes.flat):
        ax.bar(neuron_ids, mv[d], color=DIGIT_COLORS[d], width=1.0, linewidth=0)
        ax.set_title(f"Digit {d}", fontsize=9)
        ax.set_xlabel("Neuron", fontsize=7)
        if d % 5 == 0:
            ax.set_ylabel("Mean spikes", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    save(fig, "z_03_spike_rate_bars.png")


# ── plot 4 — PCA ───────────────────────────────────────────────────────────────

def plot_pca(labels, vectors):
    print("Plot 4: PCA")
    # manual PCA — no sklearn dependency required
    X   = vectors - vectors.mean(axis=0)
    cov = np.cov(X.T)
    # only keep finite values
    if not np.all(np.isfinite(cov)):
        print("  WARNING: covariance matrix has non-finite values, skipping PCA")
        return

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        print("  WARNING: eigen decomposition failed, skipping PCA")
        return

    # sort descending
    idx      = np.argsort(eigvals)[::-1]
    eigvecs  = eigvecs[:, idx]
    pc1, pc2 = eigvecs[:, 0], eigvecs[:, 1]
    proj     = X @ np.column_stack([pc1, pc2])

    var_total = eigvals.sum()
    var1 = eigvals[idx[0]] / var_total * 100 if var_total > 0 else 0
    var2 = eigvals[idx[1]] / var_total * 100 if var_total > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("PCA of spike response vectors (coloured by digit)", fontsize=13)

    for d in range(10):
        mask = labels == d
        if mask.sum() > 0:
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       color=DIGIT_COLORS[d], label=str(d),
                       alpha=0.7, s=30, edgecolors="none")

    ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)")
    ax.legend(title="Digit", markerscale=1.5, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    save(fig, "z_04_pca.png")


# ── plot 5 — t-SNE ─────────────────────────────────────────────────────────────

def plot_tsne(labels, vectors):
    print("Plot 5: t-SNE")
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  scikit-learn not installed — skipping t-SNE")
        return

    # subsample if large (t-SNE is O(n^2))
    n      = len(labels)
    max_n  = 500
    if n > max_n:
        idx    = np.random.choice(n, max_n, replace=False)
        lsub   = labels[idx]
        vsub   = vectors[idx]
    else:
        lsub, vsub = labels, vectors

    perp   = min(30, max(5, len(lsub) // 10))
    tsne   = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    proj   = tsne.fit_transform(vsub)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("t-SNE of spike response vectors (coloured by digit)", fontsize=13)

    for d in range(10):
        mask = lsub == d
        if mask.sum() > 0:
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       color=DIGIT_COLORS[d], label=str(d),
                       alpha=0.7, s=30, edgecolors="none")

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="Digit", markerscale=1.5, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    save(fig, "z_05_tsne.png")


# ── plot 6 — training dynamics ─────────────────────────────────────────────────

def plot_training_dynamics():
    print("Plot 6: training dynamics")
    if not os.path.exists(POPULATION):
        print(f"  WARNING: {POPULATION} not found — skipping")
        return

    pop = np.genfromtxt(POPULATION, delimiter=",", names=True, dtype=float,
                        encoding="utf-8")

    if pop.size == 0:
        print("  WARNING: population.csv is empty — skipping")
        return

    t               = pop["t"]
    firing_rate     = pop["firing_rate"]
    mean_membrane   = pop["mean_membrane"]
    mean_weight     = pop["mean_weight"]
    mean_pre_trace  = pop["mean_pre_trace"]
    mean_post_trace = pop["mean_post_trace"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Training dynamics over time", fontsize=13)

    axes[0].plot(t, firing_rate,     color="crimson",    linewidth=0.8)
    axes[0].set_ylabel("Firing rate")
    axes[0].set_title("Population firing rate")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, mean_membrane,   color="steelblue",  linewidth=0.8)
    axes[1].set_ylabel("Membrane (V)")
    axes[1].set_title("Mean membrane potential")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, mean_weight,     color="seagreen",   linewidth=0.8)
    axes[2].set_ylabel("Weight")
    axes[2].set_title("Mean synaptic weight")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, mean_pre_trace,  color="darkorange",  linewidth=0.8,
                 label="pre_trace")
    axes[3].plot(t, mean_post_trace, color="mediumpurple", linewidth=0.8,
                 label="post_trace")
    axes[3].set_ylabel("Trace")
    axes[3].set_xlabel("Timestep")
    axes[3].set_title("Mean STDP traces")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "z_06_training_dynamics.png")


# ── plot 7 — selectivity map ───────────────────────────────────────────────────

def plot_selectivity_map(labels, vectors, n_neurons):
    print("Plot 7: selectivity map")
    mv = mean_vectors(labels, vectors)

    # For each neuron, which digit produces the highest mean spike count?
    stack      = np.stack([mv[d] for d in range(10)], axis=0)  # (10, n_neurons)
    preferred  = np.argmax(stack, axis=0)                       # (n_neurons,)
    max_rate   = stack.max(axis=0)                              # (n_neurons,)

    # Show preferred digit as colour, masked where the neuron barely fires
    active = max_rate > max_rate.max() * 0.05
    pref_img = preferred.astype(float)
    pref_img[~active] = np.nan

    pref_2d = pref_img.reshape(28, 28)

    # custom colourmap: 10 distinct colours for digits 0-9
    cmap   = plt.cm.tab10
    bounds = np.arange(-0.5, 10.5, 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle("Per-neuron preferred digit\n"
                 "(colour = digit that drives neuron most; grey = low activity)",
                 fontsize=12)

    im = ax.imshow(pref_2d, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, ticks=range(10), label="Preferred digit")
    cbar.ax.set_yticklabels([str(d) for d in range(10)])
    ax.axis("off")

    save(fig, "z_07_selectivity_map.png")


# ── plot 8 — within-digit consistency ─────────────────────────────────────────

def plot_within_digit_consistency(labels, vectors):
    print("Plot 8: within-digit consistency")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Intra-class cosine similarity distribution per digit\n"
                 "(higher = more consistent responses)", fontsize=12)

    parts   = []
    pos     = []
    colours = []

    for d in range(10):
        mask = labels == d
        vd   = vectors[mask]
        if vd.shape[0] < 2:
            continue

        # all pairwise cosine similarities
        norms = np.linalg.norm(vd, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        vn    = vd / norms
        sim_matrix = vn @ vn.T
        # upper triangle only (exclude diagonal)
        iu    = np.triu_indices(vd.shape[0], k=1)
        sims  = sim_matrix[iu]

        parts.append(sims)
        pos.append(d)
        colours.append(DIGIT_COLORS[d])

    if not parts:
        print("  not enough data — skipping")
        plt.close(fig)
        return

    vp = ax.violinplot(parts, positions=pos, showmedians=True,
                       showextrema=True, widths=0.7)

    for body, col in zip(vp["bodies"], colours):
        body.set_facecolor(col)
        body.set_alpha(0.7)

    vp["cmedians"].set_color("black")
    vp["cmins"].set_color("black")
    vp["cmaxes"].set_color("black")
    vp["cbars"].set_color("black")

    ax.set_xticks(range(10))
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Pairwise cosine similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    save(fig, "z_08_within_digit_consistency.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {RESPONSES} ...")
    img_indices, labels, vectors, n_neurons = load_responses()
    print(f"  {len(labels)} samples | {n_neurons} neurons | "
          f"digits present: {sorted(set(labels.tolist()))}")

    plot_receptive_fields(labels, vectors, n_neurons)
    plot_similarity_matrix(labels, vectors)
    plot_spike_rate_bars(labels, vectors, n_neurons)
    plot_pca(labels, vectors)
    plot_tsne(labels, vectors)
    plot_training_dynamics()
    plot_selectivity_map(labels, vectors, n_neurons)
    plot_within_digit_consistency(labels, vectors)

    print(f"\nDone. All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
