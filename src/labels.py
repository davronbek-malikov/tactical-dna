"""
Rule-based archetype naming — assigns a soccer-domain label to each
cluster based on RANKS of its centroid stats, not hardcoded cluster
indices (KMeans cluster order is arbitrary but centroid ranks are
stable across runs with the same random_state).
"""

from __future__ import annotations
import pandas as pd

ARCHETYPE_DESCRIPTIONS = {
    "Elite Complete Forward": "Highest combined goal + assist output and highest overall attacking involvement — the Messi/Ronaldo tier.",
    "Clinical Poacher": "Highest goals & xG per 90 among high-volume shooters — efficient, high-output finishers (peak Lewandowski).",
    "Deep-Lying Playmaker": "Highest build-up involvement (xGBuildup) with modest direct output — orchestrates play from deep (Kimmich/Rakitic profile).",
    "Efficient Impact Player": "Low touch volume but the highest shot conversion rate — makes the most of limited chances.",
    "Volume Forward": "High shot and chance-creation volume at moderate efficiency — the workhorse attacking profile.",
    "Defensive Anchor": "Lowest attacking involvement, highest discipline rate — primarily defensive/holding contributors.",
}


def label_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """centroids: index=cluster id, columns=FeatureEngineer.CLUSTER_FEATURES (raw, unscaled means)."""
    c = centroids.copy()
    c["output"] = c["goals_p90"] + c["assists_p90"]

    remaining = list(c.index)
    labels: dict[int, str] = {}

    def pick(name: str, sort_col: str, ascending: bool = False):
        pool = c.loc[remaining]
        cid = pool.sort_values(sort_col, ascending=ascending).index[0]
        labels[cid] = name
        remaining.remove(cid)

    pick("Elite Complete Forward", "involvement_index")
    pick("Deep-Lying Playmaker", "xGBuildup_p90")
    pick("Clinical Poacher", "goals_p90")
    pick("Efficient Impact Player", "shot_conversion")
    pick("Defensive Anchor", "discipline_p90")
    # whatever's left is the volume/workhorse profile
    for cid in remaining:
        labels[cid] = "Volume Forward"

    return labels
