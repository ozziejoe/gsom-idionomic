"""
Plotting for the idionomic GSOM pipeline.
=========================================

Every function takes results from ``pipeline.py`` and returns a Matplotlib
figure. No global state, no ``plt.show()`` -- the caller decides whether to
render (Streamlit), save (CLI), or display (notebook). This keeps the module
usable unchanged in-browser under stlite.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import patheffects as pe

from .pipeline import shorten_feature

# tab10 palette without importing seaborn (lighter for in-browser)
_TAB10 = plt.get_cmap("tab10").colors


def _cluster_palette(levels):
    pal = {-1: (0.75, 0.75, 0.75)}
    for i, c in enumerate(levels):
        pal[int(c)] = _TAB10[i % 10]
    return pal


# ---------------------------------------------------------------- spread tuning
def plot_spread_tuning(df_spread, best_spread):
    """Spread diagnostic. The two selection criteria (QE, topology) plus map size
    (a neutral diagnostic). No clustering metric -- Step 1 is independent of Step 2.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (axes[0], "QE", "o-", "steelblue",
         "Quantization Error\n(selection · lower is better)", "Quantization Error"),
        (axes[1], "topology_rho", "^-", "forestgreen",
         "Topology Preservation\n(selection · higher is better)", "Topology ρ"),
        (axes[2], "nodes", "s-", "darkorange",
         "Map Size\n(diagnostic · not used)", "Number of Nodes"),
    ]
    for ax, col, style, color, title, ylab in panels:
        ax.plot(df_spread["spread"], df_spread[col], style, color=color, linewidth=2, markersize=8)
        ax.axvline(best_spread, color="red", linestyle="--", linewidth=2,
                   label=f"Selected: {best_spread:.2f}")
        ax.set_xlabel("Spread Factor", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    fig.suptitle("GSOM Spread Selection — by quantization error & topology only "
                 "(red = selected)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- cluster map
def plot_cluster_map(df_active, df_map):
    """Color map of participant placement; empty nodes grey, winners coloured by cluster."""
    hit_counts = df_map.groupby("output")["ID"].nunique().to_dict()
    df = df_active.copy()
    df["n_people"] = df["nodeid"].map(hit_counts).fillna(0).astype(int)

    levels = sorted(c for c in df[df["n_people"] > 0]["cluster"].unique() if c != -1)
    pal = _cluster_palette(levels)

    fig, ax = plt.subplots(figsize=(10, 7))
    empty = df[df["n_people"] == 0]
    ax.scatter(empty.x, empty.y, s=15, facecolors="none", edgecolors="grey",
               linewidth=0.8, alpha=0.4)
    winners = df[df["n_people"] > 0]

    out = winners[winners.cluster == -1]
    if not out.empty:
        ax.scatter(out.x, out.y, s=80, c=[pal[-1]], alpha=1.0, edgecolor="black",
                   linewidth=0.5, label="Outlier (-1)")
    for cl in levels:
        cd = winners[winners.cluster == cl]
        if not cd.empty:
            ax.scatter(cd.x, cd.y, s=80, c=[pal[int(cl)]], edgecolor="black",
                       linewidth=0.5, label=f"{int(cl)}")
    for _, row in winners.iterrows():
        ax.annotate(f"{int(row['n_people'])}", xy=(row["x"], row["y"]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, ha="left", va="center")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(alpha=0.2)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title("GSOM Cluster Map", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- skeleton map
def plot_skeleton_map(gsom_map, df_active, df_map, df_profiles, id_col="ID"):
    """GSOM growth skeleton: parent-child edges, seed stars, cluster ellipses."""
    hit_counts = df_map.groupby("output")["ID"].nunique().to_dict()

    df_sk = gsom_map.skeleton_dataframe()
    df_sk["x"] = pd.to_numeric(df_sk["x"], errors="coerce")
    df_sk["y"] = pd.to_numeric(df_sk["y"], errors="coerce")
    df_sk["x_int"] = np.rint(df_sk["x"]).astype("Int64")
    df_sk["y_int"] = np.rint(df_sk["y"]).astype("Int64")

    df_nodes = df_active[["nodeid", "x", "y", "cluster"]].copy()
    df_nodes["x_int"] = np.rint(df_nodes["x"]).astype(int)
    df_nodes["y_int"] = np.rint(df_nodes["y"]).astype(int)
    df_sk = df_sk.merge(df_nodes[["nodeid", "x_int", "y_int", "cluster"]],
                        on=["x_int", "y_int"], how="left")
    df_sk["n_people"] = df_sk["nodeid"].map(hit_counts).fillna(0).astype(int)

    parents = df_sk[["nodeid_tree", "x_int", "y_int"]].rename(
        columns={"nodeid_tree": "parent_id", "x_int": "parent_x", "y_int": "parent_y"})
    edges = (df_sk[["parent_id", "x_int", "y_int"]]
             .rename(columns={"x_int": "child_x", "y_int": "child_y"})
             .merge(parents, on="parent_id", how="left")
             .dropna(subset=["parent_x", "parent_y"]))

    roots = df_sk[df_sk["parent_id"].isna()]["nodeid_tree"]
    root_id = roots.iloc[0] if len(roots) else None
    seed_ids = df_sk[df_sk["parent_id"] == root_id]["nodeid_tree"].tolist() if root_id else []
    seeds = df_sk[df_sk["nodeid_tree"].isin(seed_ids)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in edges.iterrows():
        ax.plot([row["child_x"], row["parent_x"]], [row["child_y"], row["parent_y"]],
                color="grey", linewidth=0.8, alpha=0.5, zorder=1)

    empty = df_sk[df_sk["n_people"] == 0]
    ax.scatter(empty["x_int"], empty["y_int"], s=15, facecolors="none",
               edgecolors="grey", linewidth=0.8, alpha=0.4, zorder=2)

    winners = df_sk[df_sk["n_people"] > 0]
    clusters = sorted(winners["cluster"].dropna().unique())
    for cl in [c for c in clusters if c != -1]:
        cd = winners[winners.cluster == cl]
        ax.scatter(cd["x_int"], cd["y_int"], s=80, marker="o", facecolors="black",
                   edgecolors="black", linewidth=1.2, zorder=3)
        cx, cy = cd["x_int"].mean(), cd["y_int"].mean()
        if len(cd) > 1:
            width = max((cd["x_int"].max() - cd["x_int"].min()) + 1.5, 1.5)
            height = max((cd["y_int"].max() - cd["y_int"].min()) + 1.5, 1.5)
        else:
            width = height = 1.5
        ax.add_patch(Ellipse((cx, cy), width, height, facecolor="none",
                             edgecolor="black", linestyle="--", linewidth=1.0, zorder=2.5))
        n_in = len(df_profiles[df_profiles["cluster"] == cl])
        txt = ax.text(cx - width / 2 - 0.3, cy, f"C{int(cl)} (n={n_in})",
                      fontsize=9, fontweight="bold", ha="right", va="center",
                      color="black", zorder=6)
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])

    if -1 in clusters:
        out = winners[winners.cluster == -1]
        ax.scatter(out["x_int"], out["y_int"], s=80, facecolors="none",
                   edgecolors="0.4", linewidth=1.2, zorder=3)
    for _, row in winners.iterrows():
        txt = ax.annotate(f"{int(row['n_people'])}", xy=(row["x_int"], row["y_int"]),
                          xytext=(6, 0), textcoords="offset points", fontsize=8,
                          ha="left", va="center", zorder=4)
        txt.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
    if not seeds.empty:
        ax.scatter(seeds["x_int"], seeds["y_int"], s=180, marker="*",
                   facecolors="white", edgecolors="black", linewidth=1.5, zorder=5)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("GSOM Skeleton (growth edges, seed nodes ★, cluster ellipses)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- silhouette
def plot_silhouette(df_active, sil_cut):
    """Horizontal silhouette bars grouped by cluster, with the outlier cutoff."""
    df = df_active.copy().sort_values(["cluster", "silhouette"], ascending=[True, False])
    clusters = sorted(df["cluster"].unique())
    levels = [c for c in clusters if c != -1]
    pal = _cluster_palette(levels)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower, yticks, yticklabels = 0, [], []
    for cl in clusters:
        cd = df[df["cluster"] == cl]
        if len(cd) == 0:
            continue
        y_pos = np.arange(y_lower, y_lower + len(cd))
        ax.barh(y_pos, cd["silhouette"].values, height=0.8, color=pal[cl], alpha=0.85)
        label = "Outliers" if cl == -1 else f"Cluster {cl}"
        yticks.append(y_lower + len(cd) / 2)
        yticklabels.append(f"{label}\n(n={len(cd)})")
        y_lower += len(cd) + 2

    ax.axvline(x=sil_cut, color="red", linestyle="--", linewidth=2, label=f"Cutoff = {sil_cut}")
    ax.axvline(x=0, color="black", linewidth=0.5)
    valid = df_active[df_active["cluster"] != -1]["silhouette"]
    if len(valid):
        ax.axvline(x=valid.mean(), color="blue", linestyle=":", linewidth=2,
                   label=f"Mean = {valid.mean():.3f}")
    ax.set_xlabel("Silhouette Score", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_yticks(yticks); ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_xlim(-0.3, 1.0)
    ax.set_title("Cluster Fit Quality (Node Silhouettes)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- divergence
def plot_divergence(cluster_num, cluster_result, config):
    """Top-5 features where a cluster diverges most from the whole-sample mean."""
    id_col = config.id_col
    df_clusters = cluster_result.df_clusters
    whole = cluster_result.whole_sample_mean
    feature_cols = list(whole.index)

    cids = df_clusters[df_clusters["cluster"] == cluster_num][id_col].astype(str).tolist()
    cdata = cluster_result.df_profiles
    cdata = cdata[cdata["cluster"] == cluster_num]
    cmean = cdata[feature_cols].mean()
    divergence = cmean - whole
    top5 = divergence.abs().sort_values(ascending=False).head(5).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(top5))
    divs = [divergence[f] for f in top5]
    colors = ["steelblue" if d > 0 else "lightcoral" for d in divs]
    ax.barh(y_pos, divs, color=colors, edgecolor="black", linewidth=0.8, height=0.6)
    for i, f in enumerate(top5):
        d = divergence[f]
        ax.annotate(f"C:{cmean[f]:.2f} S:{whole[f]:.2f}",
                    xy=(d + (0.02 if d > 0 else -0.02), i),
                    ha="left" if d > 0 else "right", va="center", fontsize=8)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_feature(f) for f in top5], fontsize=10)
    ax.set_xlabel("Divergence from Sample Mean (Cluster - Sample)", fontsize=11)
    ax.set_title(f"Cluster {cluster_num} Divergence Profile (n={len(cids)})\n"
                 "Blue = cluster higher, Red = cluster lower",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- homogeneity
def plot_homogeneity_gain(cluster_result, config):
    """Bar chart: per-cluster mean within-cluster I^2 vs whole-sample I^2.

    A compact 'defensibility' view derived from the SE columns in the upload.
    Lower within-cluster I^2 than the whole sample = the partition reduced
    heterogeneity (the ergodic argument).
    """
    from .pipeline import _compute_i_squared
    feats = None  # resolved below
    df = cluster_result.df_full_table
    if df is None or not len(df):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No SE columns -> I-squared unavailable",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    sub = df.dropna(subset=["Sample_I2", "Cluster_I2"])
    if not len(sub):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "I-squared not computable for these features",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    g = sub.groupby("Cluster")[["Sample_I2", "Cluster_I2"]].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["Sample_I2"], width=0.4, label="Whole sample I²",
           color="lightcoral", edgecolor="black")
    ax.bar(x + 0.2, g["Cluster_I2"], width=0.4, label="Within-cluster I²",
           color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {int(c)}" for c in g.index])
    ax.set_ylabel("Mean I² (%) over important features", fontsize=11)
    ax.set_title("Homogeneity Gain: within-cluster vs whole-sample heterogeneity\n"
                 "(lower within-cluster I² = more defensible partition)",
                 fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
