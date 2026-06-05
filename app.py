"""
GSOM Idionomic Clustering -- interactive two-step app.
======================================================

Step 1 builds and tunes a Growing Self-Organizing Map of person-specific feature
matrices. Step 2 (optional) clusters the map and characterises the clusters.

Runs three ways from this one file:
  * locally / on a server:  ``streamlit run app.py``
  * fully in-browser:       via the stlite wrapper in ``index.html`` (no upload)
  * reproducibly:           the same engine powers the Colab notebook
"""

import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from gsom_idionomic import GsomConfig, build_map, cluster_map
from gsom_idionomic.demo import (
    make_sample_dataset, make_zoo_dataset, SAMPLE_RECOMMENDED, ZOO_RECOMMENDED,
)
from gsom_idionomic import plots

# Built-in sample datasets: label -> loader, recommended settings, blurb.
SAMPLES = {
    "🦓 Zoo animals (101 animals × 16 traits → animal kinds)": {
        "loader": make_zoo_dataset,
        "rec": ZOO_RECOMMENDED,
        "blurb": (
            "The classic **UCI Zoo** dataset — 101 animals described by 16 traits "
            "(hair, feathers, eggs, milk, fins, legs, …). No labels are given to "
            "the GSOM; it groups the animals purely by their traits, and they fall "
            "into natural kinds — **mammals, fish, birds, bugs & amphibians**. "
            "A nice demo that this works on *any* features, not just idionomic data."),
    },
    "🧠 Idionomic EMA (people × within-person betas → person-types, shows I²)": {
        "loader": make_sample_dataset,
        "rec": SAMPLE_RECOMMENDED,
        "blurb": (
            "80 simulated people from **four person-types** (a 2×2 of somatic vs "
            "psychosocial within-person dynamics). Includes `se_*` columns, so this "
            "one also shows the **homogeneity-gain (I²)** defensibility panel."),
    },
}

st.set_page_config(page_title="GSOM Idionomic Clustering",
                   page_icon="🗺️", layout="wide")


# --------------------------------------------------------------------- helpers
def fig_to_png_bytes(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def reset_downstream():
    """When the map is rebuilt, drop any stale clustering."""
    st.session_state["cluster"] = None


def _frange(lo, hi, step):
    vals, x = [], lo
    while x <= hi + 1e-9:
        vals.append(round(x, 2))
        x += step
    return vals


def _build_zip(m, c, cfg, se_cols):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        tables = {
            "tables/gsom_cluster_assignments.csv": c.df_clusters,
            "tables/gsom_node_quality.csv": c.df_active,
            "tables/gsom_k_evaluation.csv": c.df_k,
            "tables/gsom_characteristics.csv": c.df_characteristics,
            "tables/gsom_2x2_classification.csv": c.df_classification,
            "tables/gsom_cluster_table_full.csv": c.df_full_table,
            "tables/gsom_cluster_table_summary.csv": c.df_summary_table,
        }
        if m.df_spread is not None:
            tables["tables/gsom_spread_tuning.csv"] = m.df_spread
        for name, d in tables.items():
            zf.writestr(name, d.to_csv(index=False))
        figs = {
            "figures/gsom_skeleton_map.png":
                plots.plot_skeleton_map(m.gsom_map, c.df_active, m.df_map, c.df_profiles, cfg.id_col),
            "figures/gsom_cluster_map.png": plots.plot_cluster_map(c.df_active, m.df_map),
            "figures/gsom_silhouette.png": plots.plot_silhouette(c.df_active, cfg.sil_cut),
        }
        if m.df_spread is not None:
            figs["figures/gsom_spread_tuning.png"] = plots.plot_spread_tuning(m.df_spread, m.spread)
        if len(se_cols):
            figs["figures/gsom_homogeneity.png"] = plots.plot_homogeneity_gain(c, cfg)
        for cl in c.valid_clusters:
            figs[f"figures/gsom_cluster{cl}_divergence.png"] = plots.plot_divergence(cl, c, cfg)
        for name, fig in figs.items():
            zf.writestr(name, fig_to_png_bytes(fig))
            plt.close(fig)
        zf.writestr("gsom_params.txt",
                    f"spread={m.spread}\nbest_k={c.best_k}\n"
                    f"sil_cut={cfg.sil_cut}\nseed={cfg.seed}\n")
    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------------- state init
for key in ("data", "data_name", "map", "cluster"):
    st.session_state.setdefault(key, None)
st.session_state.setdefault("is_sample", False)
st.session_state.setdefault("sample_truth", {})
st.session_state.setdefault("sample_blurb", "")

# Widget values live in session_state under these keys so we can pre-set them
# (e.g. load the sample's recommended settings). Defaults = the real-data defaults.
SPREAD_OPTS = ["Auto (tune by QE + topology)", "Default (0.80)", "Hand-choose"]
K_OPTS = ["Auto (elbow on silhouette)", "Hand-choose"]
WIDGET_DEFAULTS = {
    "w_spread_mode": SPREAD_OPTS[0], "w_hand_spread": 0.80,
    "w_k_mode": K_OPTS[0], "w_hand_k": 4, "w_sil_cut": 0.20,
}
for _k, _v in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def set_widgets(values):
    """Pre-set widget session_state (called before the widgets are created)."""
    for k, v in values.items():
        st.session_state[k] = v


# ============================================================ SIDEBAR: data + cfg
st.sidebar.title("🗺️ GSOM Idionomic")
st.sidebar.caption("Growing Self-Organizing Map clustering for any "
                   "feature-by-ID matrix.")

st.sidebar.header("1 · Data")
up = st.sidebar.file_uploader(
    "Feature matrix CSV", type=["csv"],
    help="One row per unit. An ID column plus any numeric feature columns. "
         "(The beta_* / se_* idionomic naming is optional and unlocks the I² panel.)")
st.sidebar.caption("— or try a built-in example —")
sample_choice = st.sidebar.selectbox("Sample dataset", list(SAMPLES.keys()),
                                     label_visibility="collapsed")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("Use sample data", use_container_width=True):
    spec = SAMPLES[sample_choice]
    df_loaded = spec["loader"]()
    st.session_state.data = df_loaded
    st.session_state.data_name = df_loaded.attrs.get("name", "sample")
    st.session_state.is_sample = True
    st.session_state.sample_truth = dict(df_loaded.attrs.get("true_class", {}))
    st.session_state.sample_blurb = spec["blurb"]
    st.session_state.map = None
    reset_downstream()
    # pre-load the settings under which this sample recovers its groups cleanly
    rec = spec["rec"]
    set_widgets({"w_spread_mode": "Hand-choose", "w_hand_spread": rec["spread"],
                 "w_k_mode": "Hand-choose", "w_hand_k": rec["k"],
                 "w_sil_cut": rec["sil_cut"]})
if col_b.button("Clear", use_container_width=True):
    for k in ("data", "data_name", "map", "cluster"):
        st.session_state[k] = None
    st.session_state.is_sample = False
    st.session_state.sample_truth = {}
    set_widgets(WIDGET_DEFAULTS)

if up is not None:
    st.session_state.data = pd.read_csv(up)
    st.session_state.data_name = up.name
    st.session_state.is_sample = False
    st.session_state.sample_truth = {}
    st.session_state.map = None
    reset_downstream()

id_col = st.sidebar.text_input("ID column name", value="ID")

st.sidebar.header("Training")
training_iter = st.sidebar.slider("Growing iterations", 20, 200, 100, step=10)
smoothing_iter = st.sidebar.slider("Smoothing iterations", 10, 100, 50, step=10)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Privacy: when run from the in-browser build, your data "
                   "never leaves your computer.")


# ====================================================================== HEADER
st.title("GSOM Idionomic Clustering")
st.markdown(
    "A two-step pipeline for **any feature-by-ID matrix**: **① build** a growing "
    "self-organizing map of your rows, then optionally **② cluster** that map and "
    "characterise the groups it reveals. (Built for *idionomic* data — rows = "
    "people, features = within-person coefficients — but works on any numeric features.)")

if st.session_state.data is None:
    st.info("⬅️ Pick a **sample dataset** in the sidebar and click **Use sample "
            "data**, or upload your own CSV, to begin.")
    with st.expander("📦 Two built-in examples to try", expanded=True):
        st.markdown(
            "- **🦓 Zoo animals** — 101 animals × 16 traits. The GSOM groups them "
            "into natural kinds (mammals, fish, birds, bugs…) from traits alone. "
            "A great show of *any features by ID*.\n"
            "- **🧠 Idionomic EMA** — 80 people × within-person β coefficients in a "
            "2×2 of person-types; also demonstrates the **I²** defensibility panel.\n\n"
            "Choose one in the sidebar, click **Use sample data**, then **Build "
            "map** → **Cluster map** — its recommended settings load automatically.")
    with st.expander("What should my own CSV look like?"):
        st.markdown(
            "- One **row per unit** (person, site, case…).\n"
            "- An **`ID`** column (any name — set it in the sidebar).\n"
            "- **One or more numeric feature columns** — any names.\n\n"
            "*Optional idionomic convention:* name features "
            "`beta_X_predicts_Y` and add matching `se_*` columns to unlock the "
            "I² / homogeneity-gain panel.")
        st.dataframe(make_sample_dataset(n_per_type=2).head(4))
    st.stop()

df = st.session_state.data
beta_cols = [c for c in df.columns if c.startswith("beta_")]
se_cols = [c for c in df.columns if c.startswith("se_")]
n_feat = len(beta_cols) if beta_cols else len([c for c in df.columns
                                               if c != id_col and not c.startswith("se_")])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (IDs)", df.shape[0])
c2.metric("Features", n_feat)
c3.metric("SE columns", len(se_cols))
c4.metric("Source", st.session_state.data_name or "—")


# =================================================================== STEP 1 TAB
tab1, tab2 = st.tabs(["① Build map", "② Cluster map"])

with tab1:
    st.subheader("Step 1 — Build & tune the map")
    st.markdown(
        "This step trains the Growing Self-Organizing Map: a grid of *nodes* that "
        "grows to fit your data, placing similar rows near each other. You choose "
        "the **spread factor** (how readily the map grows), and the map is scored "
        "purely on how well it fits — quantization error and topology preservation. "
        "The result is a 2-D map of your rows. *No clustering happens here* — that's "
        "the optional Step 2.")

    if st.session_state.is_sample:
        st.success("📌 **Sample loaded** with its recommended settings pre-filled. "
                   "Just click **Build map**, then (optionally) **Cluster map** — "
                   "the natural groups come back cleanly.")
        if st.session_state.sample_blurb:
            st.caption(st.session_state.sample_blurb)

    left, right = st.columns([1, 2])
    with left:
        spread_mode = st.radio(
            "Spread factor", SPREAD_OPTS, key="w_spread_mode",
            help="Lower spread → the map grows more nodes. 'Auto' sweeps a range "
                 "and picks by a weighted blend of map-fit criteria.")
        spread_arg = "auto"
        if spread_mode == "Default (0.80)":
            spread_arg = 0.80
        elif spread_mode == "Hand-choose":
            spread_arg = st.slider("Spread value", 0.30, 0.95, step=0.05,
                                   key="w_hand_spread")

        with st.expander("Auto-tuning settings", expanded=(spread_arg == "auto")):
            srange = st.slider("Spread sweep range", 0.30, 0.95, (0.40, 0.90), step=0.05)
            sstep = st.select_slider("Sweep step", options=[0.05, 0.10], value=0.05)
            st.caption("Spread is selected from **map fit only** — an equal-weighted "
                       "blend of **quantization error** and **topology preservation**. "
                       "No clustering happens here (Step 1 is independent of Step 2).")
            w_qe = st.slider("· Quantization error", 0.0, 1.0, 0.5, 0.1)
            w_topo = st.slider("· Topology preservation", 0.0, 1.0, 0.5, 0.1)

        build_clicked = st.button("🛠️ Build map", type="primary", use_container_width=True)

    spreads = tuple(round(x, 2) for x in
                    _frange(srange[0], srange[1], sstep))
    cfg = GsomConfig(
        id_col=id_col, spread_values=spreads,
        w_qe=w_qe, w_topo=w_topo,
        training_iter=training_iter, smoothing_iter=smoothing_iter, seed=int(seed))

    if build_clicked:
        prog = st.progress(0.0, text="Starting…")

        def _cb(frac, label):
            prog.progress(min(1.0, frac), text=f"{label}…")

        try:
            with st.spinner("Training GSOM…"):
                m = build_map(df, cfg, spread=spread_arg, progress=_cb)
            st.session_state.map = m
            st.session_state.cfg = cfg
            reset_downstream()
            prog.empty()
            st.success(f"Map built — spread = {m.spread}, "
                       f"{m.n_nodes} active nodes, {m.n_winner_nodes} winner nodes.")
        except Exception as e:  # noqa
            prog.empty()
            st.error(f"Build failed: {e}")

    m = st.session_state.map
    if m is not None:
        with right:
            st.markdown(f"**Selected spread:** `{m.spread}`  ·  "
                        f"**active nodes:** {m.n_nodes}  ·  "
                        f"**winner nodes:** {m.n_winner_nodes}")
            if m.df_spread is not None:
                st.caption("**Spread diagnostic.** Auto mode tried a range of spread "
                           "values; each was scored on map fit (lower quantization "
                           "error, higher topology preservation). The red line marks "
                           "the chosen spread. Map size is shown but not used to choose.")
                show_fig(plots.plot_spread_tuning(m.df_spread, m.spread))
                with st.expander("Spread sweep table"):
                    st.dataframe(m.df_spread_scored.round(3))

        # the map itself — useful on its own, no clustering required
        st.markdown("#### The GSOM map")
        st.caption("The trained map itself — useful on its own. Each circle is a "
                   "node; nearby nodes hold similar rows. Shade = how many of your "
                   "IDs landed on a node. ★ = the original seed nodes the map grew from.")
        show_fig(plots.plot_node_map(m.gsom_map, m.df_map))

        # node locations / counts table
        st.markdown("#### Node locations & counts")
        st.caption("The underlying numbers behind the map: each node's grid "
                   "position and how many of your IDs it holds. Download below to "
                   "join node assignments back to your data.")
        node_counts = (m.df_map.groupby(["output", "x", "y"])["ID"]
                       .nunique().reset_index(name="n_people")
                       .rename(columns={"output": "nodeid"}))
        st.dataframe(node_counts, use_container_width=True, height=240)

        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Node counts (CSV)", df_to_csv_bytes(node_counts),
                           "gsom_node_counts.csv", use_container_width=True)
        d2.download_button("⬇️ Participant→node map (CSV)", df_to_csv_bytes(m.df_map),
                           "gsom_participant_map.csv", use_container_width=True)
        if m.df_spread is not None:
            d3.download_button("⬇️ Spread tuning (CSV)", df_to_csv_bytes(m.df_spread),
                               "gsom_spread_tuning.csv", use_container_width=True)
        st.info("➡️ Map built. Open the **② Cluster map** tab to partition it "
                "(optional).")


# =================================================================== STEP 2 TAB
with tab2:
    st.subheader("Step 2 — Cluster the map (optional)")
    st.markdown(
        "Now group the map into a handful of clusters. K-means runs over the **map "
        "coordinates** (not your raw features), so it groups *regions* of the map; "
        "nodes that fit their cluster poorly are flagged as outliers. You pick the "
        "number of clusters **K** and the outlier cutoff. The rest of this tab then "
        "characterises each cluster — what makes it distinct, and (for idionomic "
        "data with `se_`) how much it sharpens heterogeneity.")
    if st.session_state.map is None:
        st.warning("Build a map in Step 1 first.")
        st.stop()

    m = st.session_state.map
    cfg = st.session_state.get("cfg", GsomConfig(id_col=id_col))

    left, right = st.columns([1, 2])
    with left:
        k_mode = st.radio("Number of clusters (K)", K_OPTS, key="w_k_mode")
        k_arg = "auto"
        if k_mode == "Hand-choose":
            k_arg = st.slider("K", 2, 12, step=1, key="w_hand_k")
        k_range = st.slider("K search range (auto)", 2, 15, (2, 12))
        sil_cut = st.slider("Outlier cutoff (silhouette)", 0.0, 0.6, step=0.05,
                            key="w_sil_cut",
                            help="Map nodes below this silhouette are flagged as "
                                 "outliers (cluster −1).")
        imp = st.slider("Importance threshold (|feature value|)", 0.0, 0.6, 0.20, 0.05)
        cluster_clicked = st.button("🧩 Cluster map", type="primary",
                                    use_container_width=True)

    cfg2 = GsomConfig(
        id_col=cfg.id_col, spread_values=cfg.spread_values,
        training_iter=cfg.training_iter, smoothing_iter=cfg.smoothing_iter,
        seed=cfg.seed, k_min=k_range[0], k_max=k_range[1],
        sil_cut=sil_cut, importance_threshold=imp)

    if cluster_clicked:
        try:
            with st.spinner("Clustering…"):
                c = cluster_map(m, cfg2, k=k_arg)
            st.session_state.cluster = c
            st.session_state.cfg2 = cfg2
            st.success(f"Clustered — K = {c.best_k}, "
                       f"{len(c.valid_clusters)} clusters after outlier flagging.")
        except Exception as e:  # noqa
            st.error(f"Clustering failed: {e}")

    c = st.session_state.cluster
    if c is not None:
        cfg2 = st.session_state.get("cfg2", cfg2)
        truth = st.session_state.get("sample_truth", {})

        # Auto-name each cluster by its dominant known group (sample data only).
        # e.g. {0: "Mammal (C0)"}. For real user data there are no labels -> None.
        cluster_labels = None
        if st.session_state.is_sample and truth:
            cl_t = c.df_clusters.copy()
            cl_t["g"] = cl_t["ID"].map(truth)
            cluster_labels = {}
            for cl in c.valid_clusters:
                grp = cl_t[cl_t["cluster"] == cl]["g"].dropna()
                name = grp.value_counts().index[0] if len(grp) else f"Cluster {cl}"
                cluster_labels[int(cl)] = f"{name} (C{cl})"

        def clab(i):
            if i == -1:
                return "Outliers"
            return cluster_labels[int(i)] if cluster_labels else f"Cluster {i}"

        with right:
            counts = c.df_clusters["cluster"].value_counts().sort_index()
            counts.index = [clab(i) for i in counts.index]
            st.markdown(f"**K = {c.best_k}**  ·  clusters: "
                        f"{', '.join(str(v) for v in c.valid_clusters)}  ·  "
                        f"outliers: {int((c.df_clusters['cluster'] == -1).sum())}")
            st.bar_chart(counts)

        # For a sample with known labels: auto-names + recovery cross-tab.
        if st.session_state.is_sample and truth:
            st.markdown("#### 🏷️ Auto-named clusters")
            st.caption("Each cluster is named after its **dominant known group** "
                       "(the GSOM never saw the labels). With your own data you'd "
                       "name the clusters yourself; here it's automatic.")
            name_rows = []
            for cl in c.valid_clusters:
                grp = c.df_clusters[c.df_clusters["cluster"] == cl]["ID"].map(truth).dropna()
                top = grp.value_counts()
                name_rows.append({"Cluster": f"C{cl}",
                                  "Name": cluster_labels[int(cl)],
                                  "Dominant group": f"{top.index[0]} ({top.iloc[0]}/{len(grp)})"})
            st.dataframe(pd.DataFrame(name_rows), use_container_width=True, hide_index=True)

            xt = c.df_clusters.copy()
            xt["Known group"] = xt["ID"].map(truth)
            xt = xt.dropna(subset=["Known group"])
            xt["Cluster"] = xt["cluster"].map(clab)
            recovery = pd.crosstab(xt["Known group"], xt["Cluster"])
            st.markdown("#### ✅ Recovery check — known group × GSOM cluster")
            st.caption("Each known group landing mostly in its own cluster = the "
                       "map found the real structure from the features alone.")
            st.dataframe(recovery, use_container_width=True)

        st.markdown("#### Skeleton map with cluster topology")
        st.caption("The map's actual growth structure (parent→child edges, seed "
                   "nodes ★), with each cluster ringed and labelled. Shows how the "
                   "clusters sit relative to one another on the map.")
        show_fig(plots.plot_skeleton_map(m.gsom_map, c.df_active, m.df_map,
                                         c.df_profiles, cfg2.id_col,
                                         cluster_labels=cluster_labels))

        cmap_col, sil_col = st.columns(2)
        with cmap_col:
            st.markdown("#### Cluster map")
            st.caption("Every node coloured by its cluster; grey = empty nodes, "
                       "numbers = how many IDs are on each node.")
            show_fig(plots.plot_cluster_map(c.df_active, m.df_map,
                                            cluster_labels=cluster_labels))
        with sil_col:
            st.markdown("#### Silhouette quality")
            st.caption("How well each node sits in its cluster (higher = better). "
                       "Nodes left of the red cutoff are flagged as outliers.")
            show_fig(plots.plot_silhouette(c.df_active, cfg2.sil_cut))

        if len(se_cols):
            st.markdown("#### Homogeneity gain (I²)")
            st.caption("The idionomic 'defensibility' view: within-cluster vs "
                       "whole-sample heterogeneity (I²). Lower within-cluster I² "
                       "means the partition removed real heterogeneity — evidence "
                       "the groups are meaningful, not noise.")
            show_fig(plots.plot_homogeneity_gain(c, cfg2))

        st.markdown("#### Divergence analysis")
        st.caption("For each cluster, the features where it differs most from the "
                   "overall average — i.e. what makes that group distinctive.")
        dcols = st.columns(min(2, max(1, len(c.valid_clusters))))
        for i, cl in enumerate(c.valid_clusters):
            with dcols[i % len(dcols)]:
                lbl = cluster_labels[int(cl)] if cluster_labels else None
                show_fig(plots.plot_divergence(cl, c, cfg2, label=lbl))

        st.markdown("#### Tables")
        st.markdown(
            "Publication-ready summaries of the clusters. **Summary** = the top "
            "distinctive features per cluster for a main-text table; **Full "
            "classification** = every feature tagged *Distinctive* vs *Standard* "
            "(by importance × statistical unusualness); **Characteristics** = each "
            "cluster's biggest features by magnitude and by divergence.")
        if not len(se_cols):
            st.caption("ℹ️ I² columns are hidden — this data has no `se_` "
                       "(standard-error) columns, so heterogeneity (I²) can't be "
                       "computed. Add `se_*` columns to enable it.")

        def _show_table(df_t):
            # drop columns that are entirely empty (e.g. I² when there are no se_)
            st.dataframe(df_t.dropna(axis=1, how="all"), use_container_width=True)

        t_a, t_b, t_c = st.tabs(["Summary (journal)", "Full classification",
                                 "Characteristics"])
        with t_a:
            _show_table(c.df_summary_table)
        with t_b:
            _show_table(c.df_classification)
        with t_c:
            _show_table(c.df_characteristics)

        # -------- download everything as a ZIP --------
        st.markdown("#### Download results")
        st.caption("One ZIP with every table (CSV) and figure (PNG) from this run, "
                   "plus a parameters file — ready to drop into a paper or share.")
        zip_bytes = _build_zip(m, c, cfg2, se_cols)
        st.download_button("⬇️ Download all results (ZIP)", zip_bytes,
                           "gsom_results.zip", type="primary",
                           use_container_width=True)
