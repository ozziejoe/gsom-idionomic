# gsom-idionomic — Context for Claude

Public, citable package + app that exposes the GSOM half of the idionomic
pipeline to the world. Extracted from
`data analyses/2026 projects/pain ema study spanish group/idionomic_pipeline_template/`
(the proven, self-contained Colab notebook `GSOM_Clustering_Colab.ipynb` was the
source of truth).

## Architecture — one engine, four front doors
- `gsom_idionomic/` — installable engine.
  - `gsom.py` — the GSOM class (numpy + scipy + bigtree; `fit()` takes an optional
    `progress(frac, label)` callback for UI bars).
  - `pipeline.py` — `GsomConfig`, `build_map()` (Step 1: prepare → tune spread →
    train → map), `cluster_map()` (Step 2: K-means on map coords + silhouette
    outlier flagging + characterisation tables). Returns dataclasses of dataframes.
    **Step 1 and Step 2 are INDEPENDENT.** Spread selection uses map-fit only —
    EQUAL weight on QE + topology (`w_qe=0.5, w_topo=0.5, w_nodes=0`; map size is a
    neutral diagnostic). The sweep runs NO k-means/silhouette (`evaluate_spread`
    returns only QE/nodes/topology). Do NOT reintroduce the notebook's old
    0.3/0.3/0.4 weighting or any `best_silhouette`/`w_cluster` in Step 1 — clustering
    quality belongs to Step 2 (`cluster_map`) and is judged via within-cluster I².
  - `plots.py` — one Matplotlib figure per function, no global state (stlite-safe).
  - `demo.py` — `make_sample_dataset()`: the headline **4-type sample** (80 people,
    2×2 factorial of "somatic" × "psychosocial" signatures → types SOM/PSY/DUO/RES,
    type encoded in ID prefix). `SAMPLE_RECOMMENDED = {spread 0.6, k 4, sil_cut 0.10}`
    gives a clean **block-diagonal recovery, 0 outliers** (seed 7, noise 0.10).
    NOTE: auto-spread/auto-K does NOT reliably recover 4 here (auto-spread picks
    0.9 → over-grows); that's why the sample ships with recommended settings, which
    the app pre-loads. `make_demo_features()` (generic 3-type) kept for misc tests.
- `app.py` — Streamlit two-step wizard (tabs ① Build map / ② Cluster map). Uses
  `st.session_state` keys `data/map/cluster`; rebuilding the map nulls `cluster`.
  Widget values live under `w_*` session keys (so "Use sample data" can pre-load
  the recommended settings); set them BEFORE the widgets instantiate, never after.
  When `is_sample`, Step 2 shows a designed-type × cluster recovery cross-tab.
- `index.html` + `netlify.toml` — **stlite** in-browser build (Pyodide). Data never
  leaves the browser. Headers set COOP/COEP for Pyodide.
- `GSOM_Idionomic_Colab.ipynb` — form-driven Colab that `pip install`s the package.
- Packaging: `pyproject.toml`, `requirements.txt`, `LICENSE` (MIT), `CITATION.cff`.

## Input contract
CSV, **one row per ID, any numeric feature columns** (the tool is general — not
idionomic-only). If any col starts with `beta_`, those are the features; else every
non-ID/non-`se_` col is a feature. `se_*` cols are optional and only unlock the I²
homogeneity panel. User-facing copy was broadened (2026-06-04) to lead with the
general "feature-by-ID matrix" framing; idionomic betas are the named special case.

## Deploy (decided 2026-06-04)
**GitHub is the single source; live app via Streamlit Community Cloud (share.streamlit.io)
— one repo, one push, auto-redeploys.** Optional in-browser build via Netlify+stlite
(`index.html`, same repo, Netlify auto-deploys from GitHub). HF Docker path was tried
then dropped (HF removed Streamlit from the SDK picker; two-remote sync = more work);
Dockerfile removed but in git history if ever needed.

## Verify
`PYTHONPATH=$PWD python3 tests/test_sample_recovery.py` — fast engine test, asserts
clean block-diagonal recovery of the 4 sample types.
`PYTHONPATH=$PWD python3 tests/test_app_flow.py` — headless AppTest of the whole
flow (sample → pre-set settings → build → cluster → recovery cross-tab → download).

## Status / open items (as of 2026-06-04)
- ✅ Engine, app, Colab, packaging, and 4-type sample dataset all written and
  passing headless tests locally (Python 3.9; streamlit 1.50).
- ⬜ Not yet pushed to GitHub (local git repo initialized, 4 commits on `main`).
  Plan: publish via GitHub Desktop, then deploy on share.streamlit.io. DOI later.
  Repo slug `ozziejoe/gsom-idionomic` (GitHub user from git config).
- ⬜ **stlite `index.html` not browser-tested** (optional path) — needs a Netlify
  deploy + load test (bigtree via micropip in Pyodide; spread-sweep perf).
- ⬜ `use_container_width` triggers deprecation warnings on streamlit 1.50 (still
  works); switch to `width="stretch"` once stlite's bundled streamlit supports it.
- Possible follow-up: full homogeneity-gain plot using an uploaded
  `idiographic_results.csv` (the notebook had a richer version); current app derives
  I² from the `se_*` columns only.
