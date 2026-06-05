"""
Sample / demo feature matrices.
===============================

The headline **sample dataset** (``make_sample_dataset``) is a small, deliberately
*interpretable* idionomic dataset: 80 people from **four person-types** laid out as
a 2x2 factorial on two orthogonal "signatures" of within-person dynamics. Because
the four types occupy the four quadrants of a 2-D structure (rather than four
mutually-equidistant points, which a 2-D map cannot keep separated), the GSOM
embeds them as four distinct regions and the clustering step recovers them cleanly.

Each person's type is encoded in their ID prefix, so after clustering you can
cross-tabulate cluster vs prefix and literally see the four designed groups come back.

Variables (a daily diary / EMA study):
  predictors : Pain, Sleep (quality), Stress, Social (connection)
  outcomes   : Mood, Fatigue, PainInterfere
Feature ``beta_<predictor>_predicts_<outcome>`` is that person's within-person slope;
``se_*`` are the matching standard errors (used by the I-squared homogeneity panel).

The two orthogonal signatures:
  A "somatic"      : Pain->worse Mood, Pain->more Interference, good Sleep->less Fatigue
  B "psychosocial" : Stress->worse Mood, Stress->more Fatigue, Social->better Mood

The four types (2x2 over A and B):
  SOM  Somatic        A on,  B off   -- bodily signals drive the day
  PSY  Psychosocial   A off, B on    -- stress & connection drive the day
  DUO  Dual-burden    A on,  B on    -- both systems active
  RES  Resilient      A off, B off   -- weakly coupled ("flat") responder

``make_demo_features`` is a lighter generic generator kept for tests / quick checks.
"""

import io

import numpy as np
import pandas as pd

from ._zoo_data import ZOO_CSV


# ============================================================================
# Zoo dataset -- the headline, intuitive example (NOT idionomic).
# 101 animals x 16 traits; the GSOM groups them into animal kinds.
# ============================================================================
ZOO_FEATURES = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic",
                "predator", "toothed", "backbone", "breathes", "venomous",
                "fins", "legs", "tail", "domestic", "catsize"]

# Settings under which the GSOM cleanly groups the animals (0 outliers).
ZOO_RECOMMENDED = {"spread": 0.4, "k": 5, "sil_cut": 0.10}


def make_zoo_dataset():
    """Return the UCI Zoo feature matrix: ``ID`` (animal) + 16 trait columns.

    ``legs`` is scaled to [0, 1] so it is comparable to the boolean traits. The
    true animal class is NOT a feature; it is stashed in ``df.attrs['true_class']``
    (``{animal: class}``) for the recovery cross-tab, with ``df.attrs['legend']``
    and ``df.attrs['name']``.
    """
    raw = pd.read_csv(io.StringIO(ZOO_CSV))
    truth = dict(zip(raw["ID"], raw["class"]))
    df = raw[["ID"] + ZOO_FEATURES].copy()
    df["legs"] = df["legs"] / 8.0  # {0,2,4,5,6,8} -> [0,1]
    df.attrs["true_class"] = truth
    df.attrs["legend"] = {c: c for c in sorted(set(truth.values()))}
    df.attrs["name"] = "Zoo animals"
    return df


def zoo_truth():
    """Return {animal: class} ground truth for the zoo sample."""
    raw = pd.read_csv(io.StringIO(ZOO_CSV))
    return dict(zip(raw["ID"], raw["class"]))


PREDICTORS = ["Pain", "Sleep", "Stress", "Social"]
OUTCOMES = ["Mood", "Fatigue", "PainInterfere"]

# Two orthogonal signatures (disjoint feature sets -> independent axes).
SIGNATURE_A = {("Pain", "Mood"): -0.70, ("Pain", "PainInterfere"): +0.70,
               ("Sleep", "Fatigue"): -0.65}                         # somatic
SIGNATURE_B = {("Stress", "Mood"): -0.70, ("Stress", "Fatigue"): +0.65,
               ("Social", "Mood"): +0.70}                           # psychosocial

# 2x2 factorial: (has A, has B) and a human label.
TYPES = {
    "SOM": {"axes": (1, 0), "label": "Somatic (pain/sleep-driven)"},
    "PSY": {"axes": (0, 1), "label": "Psychosocial (stress/social-driven)"},
    "DUO": {"axes": (1, 1), "label": "Dual-burden (both)"},
    "RES": {"axes": (0, 0), "label": "Resilient (weakly coupled)"},
}

# Curated defaults: this (seed, noise, jitter) yields a clean four-cluster
# recovery with the *recommended* sample settings (spread 0.6, K=4, outlier
# cutoff 0.10). Because the shipped CSV and the GSOM training seed are both
# fixed, that recovery is fully reproducible. See tests/test_sample_recovery.py.
SAMPLE_SEED = 7
SAMPLE_NOISE = 0.10
SAMPLE_JITTER = 0.05

# Settings under which the sample recovers its four designed types cleanly.
SAMPLE_RECOMMENDED = {"spread": 0.6, "k": 4, "sil_cut": 0.10}


def _feature_names():
    return [f"{p}_predicts_{o}" for p in PREDICTORS for o in OUTCOMES]


def make_sample_dataset(n_per_type=20, seed=SAMPLE_SEED,
                        noise=SAMPLE_NOISE, jitter=SAMPLE_JITTER):
    """Return the interpretable 2x2 / four-type sample feature matrix.

    Parameters
    ----------
    n_per_type : int
        People per type (default 20 -> N=80, "smallish").
    seed : int
        RNG seed. The default is curated for a clean four-cluster recovery.
    noise : float
        Within-person noise on each person's active (strong) links.
    jitter : float
        Background noise on the inactive (near-zero) links.

    Returns
    -------
    pandas.DataFrame
        ``ID`` (type-prefixed, e.g. ``SOM03``), ``beta_*`` and ``se_*`` columns.
        ``df.attrs['legend']`` maps each prefix to its human label.
    """
    rng = np.random.default_rng(seed)
    feats = _feature_names()
    fidx = {f: j for j, f in enumerate(feats)}
    n_feat = len(feats)

    rows, se_rows, ids, labels = [], [], [], []
    for prefix, spec in TYPES.items():
        has_a, has_b = spec["axes"]
        active = {}
        if has_a:
            active.update(SIGNATURE_A)
        if has_b:
            active.update(SIGNATURE_B)
        for i in range(n_per_type):
            person = rng.normal(0, jitter, n_feat)              # near-zero background
            for (pred, out), strength in active.items():
                person[fidx[f"{pred}_predicts_{out}"]] = strength + rng.normal(0, noise)
            person = np.clip(person, -1.0, 1.0)
            se = np.clip(0.16 - 0.05 * np.abs(person) + rng.normal(0, 0.015, n_feat),
                         0.03, 0.28)
            rows.append(person)
            se_rows.append(se)
            ids.append(f"{prefix}{i + 1:02d}")
            labels.append(spec["label"])

    betas, ses = np.array(rows), np.array(se_rows)
    data = {"ID": ids}
    for j, f in enumerate(feats):
        data[f"beta_{f}"] = betas[:, j]
    for j, f in enumerate(feats):
        data[f"se_{f}"] = ses[:, j]

    df = pd.DataFrame(data)
    df.attrs["legend"] = {p: s["label"] for p, s in TYPES.items()}
    df.attrs["true_type"] = labels
    df.attrs["true_class"] = dict(zip(ids, labels))   # {ID: designed-type label}
    df.attrs["name"] = "Idionomic EMA"
    return df


def sample_legend():
    """Return {id_prefix: human label} for the sample dataset's four types."""
    return {p: s["label"] for p, s in TYPES.items()}


# ---------------------------------------------------------------- legacy/simple
def make_demo_features(n_per_type=25, n_types=3, seed=42, id_prefix="P",
                       separation=0.75, noise=0.08, n_strong=4):
    """Lighter generic generator (n_types latent groups). Kept for tests."""
    rng = np.random.default_rng(seed)
    feats = _feature_names()
    n_feat = len(feats)

    type_centres = []
    for t in range(n_types):
        centre = rng.normal(0, 0.03, n_feat)
        strong = rng.choice(n_feat, size=n_strong, replace=False)
        centre[strong] += (rng.choice([-1, 1], size=n_strong)
                           * rng.uniform(separation * 0.85, separation, size=n_strong))
        type_centres.append(centre)

    rows, se_rows, ids = [], [], []
    for t in range(n_types):
        for i in range(n_per_type):
            person = np.clip(type_centres[t] + rng.normal(0, noise, n_feat), -1.0, 1.0)
            se = np.clip(0.18 - 0.05 * np.abs(person) + rng.normal(0, 0.02, n_feat),
                         0.04, 0.30)
            rows.append(person)
            se_rows.append(se)
            ids.append(f"{id_prefix}{t + 1}{i + 1:02d}")

    betas, ses = np.array(rows), np.array(se_rows)
    data = {"ID": ids}
    for j, f in enumerate(feats):
        data[f"beta_{f}"] = betas[:, j]
    for j, f in enumerate(feats):
        data[f"se_{f}"] = ses[:, j]
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = make_sample_dataset()
    print(df.shape, "| legend:", df.attrs["legend"])
    df.to_csv("sample_data/sample_feature_matrix.csv", index=False)
    print("wrote sample_data/sample_feature_matrix.csv")
