"""
gsom-idionomic
==============

Growing Self-Organizing Map (GSOM) clustering for idionomic analysis.

Public API
----------
>>> from gsom_idionomic import GsomConfig, build_map, cluster_map, make_demo_features
>>> df = make_demo_features()
>>> cfg = GsomConfig()
>>> m = build_map(df, cfg, spread="auto")      # Step 1: build & tune the map
>>> c = cluster_map(m, cfg, k="auto")          # Step 2: cluster the map
"""

from .gsom import GSOM
from .pipeline import (
    GsomConfig,
    Features,
    MapResult,
    ClusterResult,
    prepare_features,
    tune_spread,
    pick_spread,
    build_map,
    cluster_map,
    shorten_feature,
)
from .demo import (
    make_demo_features, make_sample_dataset, sample_legend,
    make_zoo_dataset, zoo_truth,
)

__version__ = "0.1.0"

__all__ = [
    "GSOM",
    "GsomConfig",
    "Features",
    "MapResult",
    "ClusterResult",
    "prepare_features",
    "tune_spread",
    "pick_spread",
    "build_map",
    "cluster_map",
    "shorten_feature",
    "make_demo_features",
    "make_sample_dataset",
    "sample_legend",
    "make_zoo_dataset",
    "zoo_truth",
    "__version__",
]
