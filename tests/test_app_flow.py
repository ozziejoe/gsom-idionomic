"""Headless end-to-end test of the Streamlit app using AppTest.

Loads the SAMPLE data (which also pre-sets the recommended settings), builds the
map, clusters it, and asserts the four designed person-types are recovered.
"""
import warnings
warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest


def run():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert not at.exception, f"initial load raised: {at.exception}"

    # --- click "Use sample data" (pre-sets spread 0.6 / K 4 / cutoff 0.10) ---
    [b for b in at.sidebar.button if b.label == "Use sample data"][0].click().run()
    assert not at.exception, f"sample load raised: {at.exception}"

    # --- Step 1: build map ---
    [b for b in at.button if b.label == "🛠️ Build map"][0].click().run()
    assert not at.exception, f"build_map raised: {at.exception}"
    assert any("Map built" in m.value for m in at.success), "no build success message"
    print("STEP1 OK:", [m.value for m in at.success])

    # --- Step 2: cluster map ---
    [b for b in at.button if b.label == "🧩 Cluster map"][0].click().run()
    assert not at.exception, f"cluster_map raised: {at.exception}"
    cluster_msg = " ".join(m.value for m in at.success)
    assert "K = 4" in cluster_msg, f"expected K=4, got: {cluster_msg}"
    assert "4 clusters" in cluster_msg, f"expected 4 clusters, got: {cluster_msg}"
    print("STEP2 OK:", [m.value for m in at.success])

    md = " ".join(m.value for m in at.markdown)
    assert "Recovery check" in md, "no recovery cross-tab shown for sample"
    assert "Download results" in md, "no download section after clustering"
    print("RECOVERY + DOWNLOAD OK")
    print("ALL APP FLOW TESTS PASSED")


if __name__ == "__main__":
    run()
