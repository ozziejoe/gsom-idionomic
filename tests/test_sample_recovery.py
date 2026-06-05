"""Engine-level test: the sample dataset recovers its four designed types.

Fast (single build, no spread sweep). Asserts a clean block-diagonal recovery
under the sample's recommended settings.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from gsom_idionomic import GsomConfig, build_map, cluster_map
from gsom_idionomic.demo import make_sample_dataset, sample_legend, SAMPLE_RECOMMENDED


def run():
    df = make_sample_dataset()
    assert df.shape[0] == 80, df.shape
    assert sum(c.startswith("beta_") for c in df.columns) == 12
    assert sum(c.startswith("se_") for c in df.columns) == 12

    cfg = GsomConfig(sil_cut=SAMPLE_RECOMMENDED["sil_cut"])
    m = build_map(df, cfg, spread=SAMPLE_RECOMMENDED["spread"])
    c = cluster_map(m, cfg, k=SAMPLE_RECOMMENDED["k"])

    assert c.best_k == 4, c.best_k
    assert len(c.valid_clusters) == 4, c.valid_clusters
    n_out = int((c.df_clusters["cluster"] == -1).sum())
    assert n_out == 0, f"{n_out} outliers"

    # block-diagonal recovery: each designed type -> exactly one cluster
    xt = c.df_clusters.copy()
    xt["type"] = xt["ID"].str[:3].map(sample_legend())
    ct = pd.crosstab(xt["type"], xt["cluster"])
    # every row (designed type) has all 20 in a single cluster
    assert (ct.max(axis=1) == 20).all(), f"not clean:\n{ct}"
    assert (ct.sum(axis=1) == 20).all()
    # and every cluster maps to a distinct type
    assert ct.idxmax(axis=1).nunique() == 4, f"types collapsed:\n{ct}"
    print("Sample recovery clean (block-diagonal):")
    print(ct.to_string())
    print("PASSED")


if __name__ == "__main__":
    run()
