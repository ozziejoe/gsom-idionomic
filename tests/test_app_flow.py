"""Headless end-to-end test of the Streamlit app using AppTest.

Loads the default sample (Zoo, which pre-sets its recommended settings), shows
the Step-1 map without clustering, then clusters and checks recovery.
"""
import warnings
warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest


def run():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert not at.exception, f"initial load raised: {at.exception}"

    # selectbox defaults to the first sample (Zoo); load it (pre-sets spread 0.4 / K 5)
    [b for b in at.sidebar.button if b.label == "Use sample data"][0].click().run()
    assert not at.exception, f"sample load raised: {at.exception}"

    # --- Step 1: build map (the map should render with no clustering) ---
    [b for b in at.button if b.label == "🛠️ Build map"][0].click().run()
    assert not at.exception, f"build_map raised: {at.exception}"
    assert any("Map built" in m.value for m in at.success), "no build success message"
    md1 = " ".join(m.value for m in at.markdown)
    assert "The GSOM map" in md1, "Step 1 should show the map without clustering"
    print("STEP1 OK (map shown pre-clustering):", [m.value for m in at.success])

    # --- Step 2: cluster map ---
    [b for b in at.button if b.label == "🧩 Cluster map"][0].click().run()
    assert not at.exception, f"cluster_map raised: {at.exception}"
    cluster_msg = " ".join(m.value for m in at.success)
    assert "K = 5" in cluster_msg, f"expected K=5 (zoo), got: {cluster_msg}"
    assert "5 clusters" in cluster_msg, f"expected 5 clusters, got: {cluster_msg}"
    print("STEP2 OK:", [m.value for m in at.success])

    md = " ".join(m.value for m in at.markdown)
    assert "Recovery check" in md, "no recovery cross-tab shown for sample"
    assert "Download results" in md, "no download section after clustering"
    print("RECOVERY + DOWNLOAD OK")
    print("ALL APP FLOW TESTS PASSED")


if __name__ == "__main__":
    run()
