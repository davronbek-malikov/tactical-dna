"""
Player similarity — cosine similarity in scaled feature space, restricted
to players in the same predicted cluster (comparing a Poacher against
Poachers is meaningful; comparing them against Defensive Anchors isn't).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_players(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    query_vec: np.ndarray,
    cluster: int,
    feats: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    mask = (df["cluster"] == cluster).values
    pool_idx = np.where(mask)[0]
    sims = cosine_similarity(query_vec.reshape(1, -1), X_scaled[pool_idx])[0]

    result = df.iloc[pool_idx].copy()
    result["similarity"] = sims
    result = result.sort_values("similarity", ascending=False).head(top_n)
    return result[["player", "team", "league", "season", "position_group", "archetype", "similarity"] + feats]
