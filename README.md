# gsom-idionomic

**Growing Self-Organizing Map (GSOM) clustering for any feature-by-ID matrix.**

Give it **any matrix of features keyed by an ID** — one row per unit (a person, a
site, a case…), and any set of numeric feature columns — and it will:

1. **Build & tune a GSOM** of the rows (the map grows to fit the data; the spread
   factor can be auto-selected by map-fit criteria, set to a default, or hand-chosen).
2. **Cluster the map** into groups, flagging poorly-fitting nodes as outliers.
3. **Characterise** each group — divergence profiles, a 2×2 importance × unusualness
   classification, and journal-ready tables.

It was built for **idionomic** analysis — where each row is a *person* and the
features are that person's within-person AR / cross-lagged coefficients — and keeps
special support for that case: name your features `beta_*` (with matching `se_*`
standard errors) and you also get a **homogeneity-gain (I²)** view showing how much
within-group heterogeneity each cluster removes (the ergodic defensibility argument).
But nothing requires that naming — the engine clusters any numeric feature matrix.

It runs **four** ways from the **same engine**:

| Use it… | How | Notes |
|---|---|---|
| 🌐 **In your browser** | open the deployed page | Runs fully client-side via [stlite](https://github.com/whitphx/stlite) — **your data never leaves your computer.** |
| 🎛️ **As a hosted app** | one-click deploy on [Streamlit Community Cloud](https://share.streamlit.io) from this repo (or `streamlit run app.py` locally) | Same two-step wizard, server-side; auto-redeploys on every push. |
| 📓 **In Google Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ozziejoe/gsom-idionomic/blob/main/GSOM_Idionomic_Colab.ipynb) | Form-driven notebook; `pip install`s the package. |
| 🐍 **As a Python package** | `pip install gsom-idionomic` | `build_map()` / `cluster_map()` for scripts and pipelines. |

---

## Input format

A single CSV, **one row per unit** (person, site, case…):

- an **`ID`** column (any name; tell the app/`GsomConfig` what it's called, default `ID`);
- **one or more numeric feature columns** — any names you like.

That's the whole requirement. Non-numeric or missing values are coerced/zero-filled.

**Optional idionomic convention.** If your features are within-person coefficients,
name them `beta_<predictor>_predicts_<outcome>` (e.g. `beta_Pain_predicts_Mood`) and
add matching `se_<…>` standard-error columns. The `beta_`/`se_` naming gives nicer
labels and the `se_` columns unlock the **I² / homogeneity-gain** panel. Without it,
everything still works — the I² panel is simply skipped.

Don't have data handy? Click **Use sample data** in the app, or:

```python
from gsom_idionomic import make_sample_dataset
make_sample_dataset().to_csv("sample_feature_matrix.csv", index=False)
```

The bundled example ([`sample_data/sample_feature_matrix.csv`](sample_data/sample_feature_matrix.csv))
is **80 people from four interpretable person-types**, laid out as a 2×2 of two
within-person "signatures" (somatic vs psychosocial). With its recommended
settings (**spread 0.6, K 4, cutoff 0.10** — loaded automatically by *Use sample
data*) the GSOM recovers all four types as a clean block-diagonal. Details:
[`sample_data/README.md`](sample_data/README.md).

---

## Quick start — Python

```python
from gsom_idionomic import GsomConfig, build_map, cluster_map, make_sample_dataset

df  = make_sample_dataset()         # or pd.read_csv("your_feature_matrix.csv")
cfg = GsomConfig(id_col="ID")

# Step 1 — build & tune the map
m = build_map(df, cfg, spread="auto")      # or spread=0.80, or any float
print(m.spread, m.n_nodes, m.n_winner_nodes)

# Step 2 — cluster the map (optional)
c = cluster_map(m, cfg, k="auto")          # or k=4
print(c.best_k, c.valid_clusters)
c.df_clusters.to_csv("cluster_assignments.csv", index=False)
```

Figures live in `gsom_idionomic.plots` (each returns a Matplotlib figure):

```python
from gsom_idionomic import plots
plots.plot_spread_tuning(m.df_spread, m.spread)
plots.plot_skeleton_map(m.gsom_map, c.df_active, m.df_map, c.df_profiles)
plots.plot_silhouette(c.df_active, cfg.sil_cut)
plots.plot_divergence(c.valid_clusters[0], c, cfg)
plots.plot_homogeneity_gain(c, cfg)
```

## Quick start — the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then: **Use sample data** → **① Build map** → **② Cluster map** → **Download all results (ZIP)**.

---

## The two steps in detail

**① Build map.** For each candidate *spread* the GSOM is trained and its map-fit
measured by **quantization error** (fit) and **topology preservation** (Spearman ρ
between map-distance and weight-distance). **Auto mode picks the spread by an
equal-weighted blend of those two — and nothing else.** No clustering happens in
this step (the sweep runs no k-means), so building the map is fully independent of
clustering it. Outputs: the spread diagnostic (QE + topology, with map size as a
neutral diagnostic), the node-location/count table, and the seed + skeleton structure.

**② Cluster map.** K-means over the **map coordinates** (not the raw features),
with K chosen by an elbow on the silhouette curve (or set by hand). Nodes below
the silhouette **outlier cutoff** (default 0.20) become cluster −1. Then each
cluster is characterised: divergence from the sample mean, a 2×2
importance × unusualness classification, summary/full journal tables, and — when
`se_*` standard-error columns are present (the idionomic case) — the within-cluster
vs whole-sample I² that quantifies how defensible the partition is.

---

## Deploying the live app (recommended: Streamlit Community Cloud)

The simplest setup is **one repo, one sync**: this GitHub repo is the single source,
and [Streamlit Community Cloud](https://share.streamlit.io) runs the live app from it.

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub → **New app**.
3. Pick this repo, branch `main`, main file `app.py` → **Deploy**.

It installs `requirements.txt` and runs `app.py`. **Every push auto-redeploys** — no
second upload to keep in sync. (The free tier sleeps after inactivity and wakes on
the next visit.)

### Optional: fully in-browser build (Netlify + stlite)

For a **zero-upload, runs-entirely-in-the-browser** version (privacy: data never
leaves the user's machine), the repo also ships `index.html` + `netlify.toml`. Point
Netlify at the repo root (no build command); `index.html` boots
[stlite](https://github.com/whitphx/stlite), loading `app.py` + `gsom_idionomic/`
into an in-browser Python runtime.

> First load pulls ~30 MB of the scientific Python stack into the browser
> (~20–40 s, then cached). Slower than native but fine at typical scales.

---

## Install for development

```bash
git clone https://github.com/ozziejoe/gsom-idionomic
cd gsom-idionomic
pip install -e .
python tests/test_app_flow.py        # headless end-to-end check
```

## Citing

If you use this in research, please cite it (see [`CITATION.cff`](CITATION.cff)).
A Zenodo DOI will be minted from the first GitHub release.

## License

MIT — see [`LICENSE`](LICENSE).
