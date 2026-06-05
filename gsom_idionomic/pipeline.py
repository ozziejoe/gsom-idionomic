"""
Idionomic GSOM pipeline orchestration.
======================================

Two public entry points mirror the two-step workflow:

* ``build_map(...)``   -- Step 1: prepare features, (optionally) tune the spread
  factor, train the final GSOM, and map every participant to a node.
* ``cluster_map(...)`` -- Step 2: k-means over the map coordinates with
  silhouette-based outlier flagging, then cluster characterisation tables.

Both return plain dataclasses of dataframes/arrays so the same results feed the
Streamlit app, the Colab notebook, and any downstream script. Plotting lives in
``plots.py`` and is deliberately kept separate from computation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import spearmanr, ttest_1samp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

from .gsom import GSOM


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class GsomConfig:
    """All tunable settings, with defaults matching the IMPACT idionomic pipeline."""

    id_col: str = "ID"
    # spread tuning
    spread_values: tuple = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                            0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    # Step 1 (build map) is independent of Step 2 (cluster map). Spread is
    # selected from MAP-FIT criteria only: an equal-weighted blend of quantization
    # error and topology preservation. Map size (nodes) is a neutral diagnostic
    # (weight 0). No clustering metric enters here.
    w_qe: float = 0.5
    w_topo: float = 0.5
    w_nodes: float = 0.0
    # training
    training_iter: int = 100
    smoothing_iter: int = 50
    seed: int = 42
    # clustering
    k_min: int = 2
    k_max: int = 12
    sil_cut: float = 0.20
    # characterisation
    importance_threshold: float = 0.20
    unusualness_alpha: float = 0.05
    use_sparse: bool = False


# ----------------------------------------------------------------------------
# Feature preparation
# ----------------------------------------------------------------------------
def _clean_column_name(col, id_col):
    if col == id_col:
        return col
    s = str(col)
    if s.startswith("beta_"):
        s = s[5:]
    s = s.replace("_PSD_predicts_", "_predicts_")
    s = s.replace("_PSD", "")
    return s


def _shorten_var(v, max_len=8):
    v = v.replace("Interfer", "").replace("_PSD", "")
    return v[:max_len] if len(v) > max_len else v


def shorten_feature(f):
    """Compact label for a 'X_predicts_Y' feature, e.g. 'pain -> sleep'."""
    parts = f.split("_predicts_")
    if len(parts) == 2:
        return f"{_shorten_var(parts[0])} -> {_shorten_var(parts[1])}"
    return f[:18]


@dataclass
class Features:
    df_clean: pd.DataFrame            # ID + cleaned feature columns
    feature_cols: list
    data_matrix: np.ndarray
    se_map: dict                      # feature_col -> se column in raw frame
    df_raw: pd.DataFrame              # original upload (for I-squared via SEs)
    n_missing: int = 0


def prepare_features(df_features, config=None):
    """Identify ID/beta/SE columns, clean names, fill NAs, build the matrix."""
    config = config or GsomConfig()
    id_col = config.id_col
    if id_col not in df_features.columns:
        raise ValueError(
            f"ID column '{id_col}' not found. Columns: {list(df_features.columns)[:8]}..."
        )

    beta_cols = [c for c in df_features.columns if c.startswith("beta_")]
    if len(beta_cols) == 0:
        beta_cols = [c for c in df_features.columns
                     if c != id_col and not c.startswith("se_")]

    df_clean = df_features[[id_col] + beta_cols].copy()
    df_clean.columns = [_clean_column_name(c, id_col) for c in df_clean.columns]
    df_clean[id_col] = df_clean[id_col].astype(str)
    feature_cols = [c for c in df_clean.columns if c != id_col]

    n_missing = int(df_clean[feature_cols].isna().sum().sum())
    if n_missing > 0:
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)

    # map each cleaned feature back to an SE column if one exists
    se_map = {}
    for feat in feature_cols:
        for cand in (f"se_{feat}", f"se_beta_{feat}", f"se_{feat}_PSD"):
            if cand in df_features.columns:
                se_map[feat] = cand
                break

    df_raw = df_features.copy()
    df_raw[id_col] = df_raw[id_col].astype(str)

    return Features(
        df_clean=df_clean,
        feature_cols=feature_cols,
        data_matrix=df_clean[feature_cols].values,
        se_map=se_map,
        df_raw=df_raw,
        n_missing=n_missing,
    )


# ----------------------------------------------------------------------------
# Node helpers
# ----------------------------------------------------------------------------
def _mappoint_id_mapping(map_points, id_col="ID"):
    rows = []
    for row in map_points.itertuples():
        ids = getattr(row, id_col) if hasattr(row, id_col) else row.ID
        for x in ids:
            rows.append([x, row.x, row.y, row.output, row.hit_count])
    return pd.DataFrame(rows, columns=["ID", "x", "y", "output", "hit_count"])


def _active_node_mask(gsom_map):
    coords = np.asarray(gsom_map.node_coordinate, dtype=float)
    weights = np.asarray(gsom_map.node_list, dtype=float)
    mask = (np.isfinite(coords).all(axis=1)
            & np.isfinite(weights).all(axis=1)
            & (weights.var(axis=1) > 0))
    return mask, coords, weights


# ----------------------------------------------------------------------------
# Step 1a: spread tuning
# ----------------------------------------------------------------------------
def evaluate_spread(spread_factor, df_clean, config):
    """Fit a GSOM at one spread and return the 4 quality criteria."""
    id_col = config.id_col
    np.random.seed(config.seed)
    data_train = df_clean.drop(columns=[id_col])

    gsom_map = GSOM(spread_factor, data_train.shape[1])
    gsom_map.fit(data_train.to_numpy(), config.training_iter, config.smoothing_iter)

    map_points = gsom_map.predict(df_clean, id_col)
    df_map = _mappoint_id_mapping(map_points.copy(), id_col)

    # 1: Quantization error
    df_weights = pd.DataFrame(gsom_map.node_list)
    df_weights = df_map[["ID", "output", "x", "y"]].merge(
        df_weights, left_on="output", right_index=True)
    df_input = df_clean.sort_values(by=[id_col]).drop(columns=[id_col]).reset_index(drop=True)
    df_result = (df_weights.sort_values(by=["ID"])
                 .drop(columns=["ID", "output", "x", "y"])
                 .apply(pd.to_numeric).reset_index(drop=True))
    QE = float(np.mean([euclidean(df_input.iloc[i], df_result.iloc[i])
                        for i in range(len(df_result))]))

    # 2: active node count (map size -- neutral diagnostic only)
    mask, coords, weights = _active_node_mask(gsom_map)
    n_nodes = int(mask.sum())
    coords_active = coords[mask]
    weights_active = weights[mask]

    # 3: topology preservation
    if n_nodes >= 3:
        D_coords = squareform(pdist(coords_active, metric="euclidean"))
        D_weights = squareform(pdist(weights_active, metric="euclidean"))
        tri = np.triu_indices_from(D_coords, k=1)
        topology_rho, _ = spearmanr(D_coords[tri], D_weights[tri])
    else:
        topology_rho = np.nan

    # Step 1 (building the map) is independent of Step 2 (clustering it): the
    # spread sweep deliberately does NOT run any k-means / silhouette. Only the
    # two map-fit criteria (QE, topology) drive spread selection; map size is a
    # neutral diagnostic. Cluster quality is assessed later, in cluster_map().
    return {"spread": spread_factor, "QE": QE, "nodes": n_nodes,
            "topology_rho": topology_rho}


def tune_spread(df_clean, config, progress=None):
    """Sweep all spread values; return a dataframe of criteria per spread."""
    rows = []
    spreads = list(config.spread_values)
    for i, s in enumerate(spreads):
        rows.append(evaluate_spread(s, df_clean, config))
        if progress:
            progress((i + 1) / len(spreads), f"spread {s:.2f}")
    return pd.DataFrame(rows)


def pick_spread(df_spread, config):
    """Select the spread from map-fit criteria only. Returns (best, scored).

    Score = w_qe * QE_norm + w_topo * topology_norm (defaults 0.5 / 0.5). Map size
    (nodes) is a neutral diagnostic, weight 0 by default. No clustering metric is
    involved -- Step 1 is independent of Step 2.
    """
    r = df_spread.copy().sort_values("spread").reset_index(drop=True)
    eps = 1e-12
    r["qe_n"] = 1 - (r["QE"] - r["QE"].min()) / (r["QE"].max() - r["QE"].min() + eps)
    r["nodes_n"] = 1 - (r["nodes"] - r["nodes"].min()) / (r["nodes"].max() - r["nodes"].min() + eps)
    r["topo_n"] = (r["topology_rho"] - r["topology_rho"].min()) / (r["topology_rho"].max() - r["topology_rho"].min() + eps)
    r["score"] = (config.w_qe * r["qe_n"] + config.w_topo * r["topo_n"]
                  + config.w_nodes * r["nodes_n"])
    best = float(r.loc[r["score"].idxmax(), "spread"])
    return best, r


# ----------------------------------------------------------------------------
# Step 1b: train final map
# ----------------------------------------------------------------------------
@dataclass
class MapResult:
    gsom_map: object
    spread: float
    df_map: pd.DataFrame              # one row per participant: ID, x, y, output, hit_count
    df_spread: pd.DataFrame = None    # spread tuning criteria (if auto)
    df_spread_scored: pd.DataFrame = None
    features: Features = None
    n_nodes: int = 0
    n_winner_nodes: int = 0


def build_map(df_features, config=None, spread="auto", progress=None):
    """Step 1. Prepare features, choose spread, train GSOM, map participants.

    Parameters
    ----------
    spread : 'auto' | float
        'auto' runs the spread sweep + multi-criteria selection. A float trains
        directly at that spread (the 'default 0.8' or 'hand-chosen' options).
    """
    config = config or GsomConfig()
    feats = prepare_features(df_features, config)

    df_spread = df_spread_scored = None
    if spread == "auto":
        if progress:
            progress(0.0, "tuning spread")
        df_spread = tune_spread(feats.df_clean, config, progress=progress)
        spread_value, df_spread_scored = pick_spread(df_spread, config)
    else:
        spread_value = float(spread)

    np.random.seed(config.seed)
    gsom_map = GSOM(spread_value, feats.data_matrix.shape[1])
    gsom_map.fit(feats.data_matrix, config.training_iter, config.smoothing_iter,
                 progress=(lambda f, lbl: progress(f, lbl)) if progress else None)

    map_points = gsom_map.predict(feats.df_clean, config.id_col)
    df_map = _mappoint_id_mapping(map_points.copy(), config.id_col)

    mask, _, _ = _active_node_mask(gsom_map)
    return MapResult(
        gsom_map=gsom_map, spread=spread_value, df_map=df_map,
        df_spread=df_spread, df_spread_scored=df_spread_scored,
        features=feats, n_nodes=int(mask.sum()),
        n_winner_nodes=df_map[["x", "y"]].drop_duplicates().shape[0],
    )


# ----------------------------------------------------------------------------
# Step 2: cluster the map
# ----------------------------------------------------------------------------
def _elbow_pick_on_silhouette(k_eval):
    def n01(a):
        a = np.asarray(a, dtype=float)
        return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)
    r = k_eval.sort_values("K").reset_index(drop=True)
    x, y = n01(r["K"].to_numpy()), n01(r["Silhouette"].to_numpy())
    A, B = y[0] - y[-1], x[-1] - x[0]
    C = x[0] * y[-1] - x[-1] * y[0]
    dist = np.abs(A * x + B * y + C) / np.sqrt(A * A + B * B + 1e-12)
    return int(r.loc[int(np.argmax(dist)), "K"])


def _compute_i_squared(df_raw, ids_list, feat, se_col, id_col):
    beta_col = f"beta_{feat}" if f"beta_{feat}" in df_raw.columns else feat
    if beta_col not in df_raw.columns or se_col not in df_raw.columns:
        return np.nan
    subset = df_raw[df_raw[id_col].astype(str).isin([str(i) for i in ids_list])]
    if len(subset) < 2:
        return np.nan
    betas = subset[beta_col].values.astype(float)
    ses = subset[se_col].values.astype(float)
    valid = ~(np.isnan(betas) | np.isnan(ses) | (ses <= 0))
    betas, ses = betas[valid], ses[valid]
    if len(betas) < 2:
        return np.nan
    w = 1 / (ses ** 2)
    wm = np.sum(w * betas) / np.sum(w)
    Q = np.sum(w * (betas - wm) ** 2)
    dfree = len(betas) - 1
    return max(0.0, ((Q - dfree) / Q) * 100) if Q > dfree else 0.0


@dataclass
class ClusterResult:
    best_k: int
    df_k: pd.DataFrame                # K evaluation (silhouette, DBI)
    df_active: pd.DataFrame           # every node: nodeid, x, y, cluster, silhouette
    df_clusters: pd.DataFrame         # every participant: ID, cluster, x, y
    df_profiles: pd.DataFrame         # df_clean + cluster
    cluster_means: pd.DataFrame
    whole_sample_mean: pd.Series
    valid_clusters: list
    df_characteristics: pd.DataFrame  # top-5 magnitude + divergence per cluster
    df_classification: pd.DataFrame   # 2x2 importance x unusualness
    df_full_table: pd.DataFrame       # journal full table
    df_summary_table: pd.DataFrame    # journal summary table
    labels_all: np.ndarray = None
    active_idx: np.ndarray = None


def cluster_map(map_result, config=None, k="auto", progress=None):
    """Step 2. K-means over map coords + outlier flagging + characterisation."""
    config = config or GsomConfig()
    id_col = config.id_col
    gsom_map = map_result.gsom_map
    df_map = map_result.df_map
    feats = map_result.features
    df_clean = feats.df_clean
    feature_cols = feats.feature_cols

    mask, coords, weights = _active_node_mask(gsom_map)
    active_idx = np.nonzero(mask)[0]
    coords_active = coords[mask]

    winner_set = set((int(x), int(y))
                     for x, y in df_map[["x", "y"]].drop_duplicates().values)
    hit_mask = np.array([(int(coords[idx, 0]), int(coords[idx, 1])) in winner_set
                         for idx in active_idx])
    X_probe = coords_active[hit_mask, :2]
    X_all = coords_active[:, :2]

    # --- K selection ---
    k_max = min(config.k_max, len(X_probe) - 1)
    k_rows = []
    for kk in range(config.k_min, k_max + 1):
        km = KMeans(n_clusters=kk, n_init="auto", random_state=config.seed).fit(X_probe)
        k_rows.append({"K": kk, "Silhouette": silhouette_score(X_probe, km.labels_),
                       "DBI": davies_bouldin_score(X_probe, km.labels_)})
    df_k = pd.DataFrame(k_rows)
    best_k = int(k) if k != "auto" else _elbow_pick_on_silhouette(df_k)
    if progress:
        progress(0.5, f"K={best_k}")

    # --- final clustering with outlier flagging ---
    km_final = KMeans(n_clusters=best_k, n_init="auto", random_state=config.seed).fit(X_probe)
    labels_all = km_final.predict(X_all).astype(int)

    sil_vals = (silhouette_samples(X_all, labels_all)
                if np.unique(labels_all).size >= 2 else np.zeros(len(X_all)))

    # node -> cluster (with silhouette outliers flagged as -1)
    node_cluster = {}
    node_sil = {}
    for i, idx in enumerate(active_idx):
        cl = int(labels_all[i])
        if sil_vals[i] < config.sil_cut:
            cl = -1
        node_cluster[idx] = cl
        node_sil[idx] = float(sil_vals[i])

    # participant clusters via their winning node
    part_rows = []
    for _, row in df_map.iterrows():
        node_idx = int(row["output"])
        part_rows.append({id_col: row["ID"],
                          "cluster": node_cluster.get(node_idx, -1),
                          "x": int(row["x"]), "y": int(row["y"])})
    df_clusters = pd.DataFrame(part_rows)

    # df_active over ALL map nodes (for skeleton/cluster maps)
    node_rows = []
    for (x, y), idx in gsom_map.map.items():
        node_rows.append({"nodeid": idx, "x": x, "y": y,
                          "cluster": node_cluster.get(idx, -1),
                          "silhouette": node_sil.get(idx, 0.0)})
    df_active = pd.DataFrame(node_rows)

    # profiles
    df_profiles = df_clean.merge(df_clusters[[id_col, "cluster"]], on=id_col, how="inner")
    cluster_means = df_profiles.groupby("cluster")[feature_cols].mean()
    whole_sample_mean = df_clean[feature_cols].mean()
    valid_clusters = sorted([c for c in df_clusters["cluster"].unique() if c != -1])

    # --- characterisation tables ---
    df_char, df_class, df_full, df_summary = _characterise(
        config, feats, df_clusters, cluster_means, whole_sample_mean,
        valid_clusters, df_clean, feature_cols)

    if progress:
        progress(1.0, "done")

    return ClusterResult(
        best_k=best_k, df_k=df_k, df_active=df_active, df_clusters=df_clusters,
        df_profiles=df_profiles, cluster_means=cluster_means,
        whole_sample_mean=whole_sample_mean, valid_clusters=valid_clusters,
        df_characteristics=df_char, df_classification=df_class,
        df_full_table=df_full, df_summary_table=df_summary,
        labels_all=labels_all, active_idx=active_idx,
    )


def _characterise(config, feats, df_clusters, cluster_means, whole_sample_mean,
                  valid_clusters, df_clean, feature_cols):
    id_col = config.id_col
    df_raw = feats.df_raw
    se_map = feats.se_map

    # sample-level I-squared per feature
    all_ids = df_clean[id_col].tolist()
    sample_i2 = {f: (_compute_i_squared(df_raw, all_ids, f, se_map[f], id_col)
                     if f in se_map else np.nan) for f in feature_cols}

    char_rows, class_rows, full_rows = [], [], []
    IMP, ALPHA = config.importance_threshold, config.unusualness_alpha

    for cl in valid_clusters:
        cids = df_clusters[df_clusters["cluster"] == cl][id_col].tolist()
        cdata = df_clean[df_clean[id_col].isin([str(i) for i in cids])]
        cmean = cdata[feature_cols].mean()
        n_c = len(cids)
        divergence = cmean - whole_sample_mean
        cluster_i2 = {f: (_compute_i_squared(df_raw, cids, f, se_map[f], id_col)
                          if f in se_map else np.nan) for f in feature_cols}

        def _row(feat, table):
            c_b, s_b = cmean[feat], whole_sample_mean[feat]
            s_i2, c_i2 = sample_i2.get(feat, np.nan), cluster_i2.get(feat, np.nan)
            delta = c_i2 - s_i2 if not (np.isnan(s_i2) or np.isnan(c_i2)) else np.nan
            return {"Cluster": cl, "n": n_c, "Table": table,
                    "Predictor": shorten_feature(feat),
                    "Cluster_Beta": round(c_b, 3), "Sample_Beta": round(s_b, 3),
                    "Difference": round(c_b - s_b, 3),
                    "Sample_I2": None if np.isnan(s_i2) else round(s_i2, 1),
                    "Cluster_I2": None if np.isnan(c_i2) else round(c_i2, 1),
                    "Delta_I2": None if np.isnan(delta) else round(delta, 1)}

        for feat in cmean.abs().sort_values(ascending=False).head(5).index:
            char_rows.append(_row(feat, "Magnitude"))
        for feat in divergence.abs().sort_values(ascending=False).head(5).index:
            char_rows.append(_row(feat, "Divergence"))

        # 2x2 classification + journal table over all features
        for feat in feature_cols:
            c_b, s_b = cmean[feat], whole_sample_mean[feat]
            diff = c_b - s_b
            is_important = (abs(c_b) > IMP) or (abs(s_b) > IMP)
            vals = cdata[feat].dropna()
            if len(vals) >= 2:
                _, p = ttest_1samp(vals, s_b)
            else:
                p = np.nan
            is_unusual = (not np.isnan(p)) and (p < ALPHA)
            higher = c_b > s_b

            if is_important and is_unusual and higher:
                cat = "Distinctive_Driver_HIGH"
            elif is_important and is_unusual and not higher:
                cat = "Distinctive_NonDriver_LOW"
            elif is_important and not is_unusual:
                cat = "Standard_Driver"
            else:
                cat = "Non_Factor"
            class_rows.append({"cluster": cl, "feature": shorten_feature(feat),
                               "cluster_beta": round(c_b, 3), "sample_beta": round(s_b, 3),
                               "difference": round(diff, 3),
                               "p_value": None if np.isnan(p) else round(p, 4),
                               "category": cat})

            if is_important:
                if is_unusual and diff > 0:
                    jcat = "Distinctive (+)"
                elif is_unusual and diff < 0:
                    jcat = "Distinctive (-)"
                else:
                    jcat = "Standard"
                s_i2, c_i2 = sample_i2.get(feat, np.nan), cluster_i2.get(feat, np.nan)
                delta = c_i2 - s_i2 if not (np.isnan(s_i2) or np.isnan(c_i2)) else np.nan
                full_rows.append({"Cluster": cl, "n": n_c, "Category": jcat,
                                  "Predictor": shorten_feature(feat),
                                  "Cluster_beta": round(c_b, 2), "Sample_beta": round(s_b, 2),
                                  "Diff": round(diff, 2),
                                  "p": None if np.isnan(p) else round(p, 3),
                                  "Sample_I2": None if np.isnan(s_i2) else round(s_i2, 0),
                                  "Cluster_I2": None if np.isnan(c_i2) else round(c_i2, 0),
                                  "Delta_I2": None if np.isnan(delta) else round(delta, 0),
                                  "_abs_beta": abs(c_b), "_abs_diff": abs(diff)})

    df_char = pd.DataFrame(char_rows)
    df_class = pd.DataFrame(class_rows)
    df_full = pd.DataFrame(full_rows)

    # full + summary journal tables
    if len(df_full):
        cat_order = {"Distinctive (+)": 0, "Distinctive (-)": 1, "Standard": 2}
        df_full["_cat_order"] = df_full["Category"].map(cat_order)
        df_full_sorted = df_full.sort_values(
            ["Cluster", "_cat_order", "_abs_diff"], ascending=[True, True, False])
        df_full_export = df_full_sorted.drop(columns=["_abs_beta", "_abs_diff", "_cat_order"])

        TOP_N = 2
        summary = []
        for cl in valid_clusters:
            cdf = df_full[df_full["Cluster"] == cl]
            for catname, sortcol in (("Distinctive (+)", "_abs_diff"),
                                     ("Standard", "_abs_beta"),
                                     ("Distinctive (-)", "_abs_diff")):
                sub = cdf[cdf["Category"] == catname].sort_values(sortcol, ascending=False).head(TOP_N)
                summary.extend(sub.to_dict("records"))
        cols = ["Cluster", "n", "Category", "Predictor", "Cluster_beta", "Sample_beta",
                "Diff", "p", "Sample_I2", "Cluster_I2", "Delta_I2"]
        df_summary = pd.DataFrame(summary)[cols] if summary else pd.DataFrame(columns=cols)
    else:
        df_full_export = pd.DataFrame()
        df_summary = pd.DataFrame()

    return df_char, df_class, df_full_export, df_summary
