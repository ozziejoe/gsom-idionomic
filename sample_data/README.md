# Sample datasets

Two built-in examples ship with the app (sidebar → **Sample dataset**).

## 1. `zoo_feature_matrix.csv` — the headline example 🦓

The classic **UCI Zoo** dataset: **101 animals × 16 traits** (`hair`, `feathers`,
`eggs`, `milk`, `airborne`, `aquatic`, `predator`, `toothed`, `backbone`,
`breathes`, `venomous`, `fins`, `legs`, `tail`, `domestic`, `catsize`). `legs` is
scaled to [0, 1]; everything else is 0/1. `ID` is the animal name.

The GSOM is given **no labels** — it groups animals purely by their traits, and
they fall into natural kinds. With the recommended settings (**spread 0.4, K 5,
outlier cutoff 0.10**) you get a clean grouping with **0 outliers**: all 41
mammals in one cluster, all fish in another, birds, and a "bugs & amphibians"
group. The true animal class for each ID is in
[`zoo_classes.csv`](zoo_classes.csv) (used only for the recovery cross-tab; it is
**not** a feature). This example shows the tool works on *any* feature-by-ID
matrix — not just idionomic data.

Source: UCI Machine Learning Repository, *Zoo* data set (R. Forsyth), public domain.

## 2. `sample_feature_matrix.csv` — idionomic example 🧠

A small, **interpretable** idionomic feature matrix: **80 simulated people** from
**four person-types**, designed so the GSOM pipeline cleanly recovers the groups.

## Shape
- 80 rows (one per person), `ID` + 12 `beta_*` + 12 `se_*` columns.
- Imagine a daily-diary / EMA study with within-person regressions of each
  **outcome** on each **predictor**; `beta_X_predicts_Y` is that person's slope and
  `se_X_predicts_Y` its standard error.

| | |
|---|---|
| **Predictors** | Pain, Sleep (quality), Stress, Social (connection) |
| **Outcomes** | Mood, Fatigue, PainInterfere |

## The four person-types (encoded in the `ID` prefix)

The types are a **2×2 factorial** over two orthogonal "signatures":

- **Signature A — somatic:** Pain→worse Mood, Pain→more Pain-Interference, good Sleep→less Fatigue.
- **Signature B — psychosocial:** Stress→worse Mood, Stress→more Fatigue, Social→better Mood.

| Prefix | Type | Signature A | Signature B |
|---|---|:---:|:---:|
| `SOM` | Somatic (pain/sleep-driven) | ✅ | — |
| `PSY` | Psychosocial (stress/social-driven) | — | ✅ |
| `DUO` | Dual-burden | ✅ | ✅ |
| `RES` | Resilient (weakly coupled / "flat" responder) | — | — |

20 people per type. Laying the types out as a 2×2 (rather than four
mutually-equidistant points, which a 2-D map cannot keep apart) is what lets the
GSOM embed them as four separable regions.

## Recommended settings (clean recovery)

In the app: click **Use sample data** — the recommended settings load
automatically. They are:

- **spread = 0.6**, **K = 4**, **outlier cutoff = 0.10** (training 100 / smoothing 50, seed 42).

Under these, every designed type maps to its own cluster (a clean block-diagonal
recovery, 0 outliers). Regenerate the file with
`python -m gsom_idionomic.demo` or `gsom_idionomic.make_sample_dataset()`.
