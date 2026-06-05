# Sample dataset — `sample_feature_matrix.csv`

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
