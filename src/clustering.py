"""
Clustering algorithm bake-off.

Runs K-Means, Agglomerative (Ward), DBSCAN, and Gaussian Mixture across a
range of k / eps, scores each with three independent internal-validation
metrics (no ground truth exists — these are unlabeled players), and
returns a leaderboard so the winner is picked by evidence, not by default.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

K_RANGE = range(3, 9)


def _score(X: np.ndarray, labels: np.ndarray) -> dict | None:
    mask = labels != -1  # exclude DBSCAN noise from scoring
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2 or n_clusters >= len(X):
        return None
    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "n_clusters": n_clusters,
        "noise_pct": float((~mask).mean()),
    }


def suggest_dbscan_eps(X: np.ndarray, min_samples: int) -> list[float]:
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nn.kneighbors(X)
    kth = np.sort(dists[:, -1])
    return [float(np.percentile(kth, p)) for p in (75, 85, 90, 95)]


def run_bakeoff(X: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    rows = []

    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        s = _score(X, labels)
        if s:
            rows.append({"algorithm": "KMeans", "params": f"k={k}", **s})

    for k in K_RANGE:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        s = _score(X, labels)
        if s:
            rows.append({"algorithm": "Agglomerative", "params": f"k={k}", **s})

    for k in K_RANGE:
        model = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
        labels = model.fit_predict(X)
        s = _score(X, labels)
        if s:
            rows.append({"algorithm": "GaussianMixture", "params": f"k={k}", **s})

    for min_samples in (10, 15, 20):
        for eps in suggest_dbscan_eps(X, min_samples):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            s = _score(X, labels)
            if s and 2 <= s["n_clusters"] <= 12:
                rows.append({
                    "algorithm": "DBSCAN",
                    "params": f"eps={eps:.2f},min_samples={min_samples}",
                    **s,
                })

    board = pd.DataFrame(rows)
    # Composite rank: higher silhouette + calinski_harabasz is better,
    # lower davies_bouldin is better. Rank each, average the ranks.
    board["rank_sil"] = board["silhouette"].rank(ascending=False)
    board["rank_dbi"] = board["davies_bouldin"].rank(ascending=True)
    board["rank_ch"] = board["calinski_harabasz"].rank(ascending=False)
    board["composite_rank"] = board[["rank_sil", "rank_dbi", "rank_ch"]].mean(axis=1)
    return board.sort_values("composite_rank").reset_index(drop=True)


def fit_final(algorithm: str, params: dict, X: np.ndarray, random_state: int = 42):
    if algorithm == "KMeans":
        model = KMeans(n_clusters=params["k"], n_init=10, random_state=random_state)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=params["k"], linkage="ward")
    elif algorithm == "GaussianMixture":
        model = GaussianMixture(n_components=params["k"], random_state=random_state, n_init=3)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    else:
        raise ValueError(algorithm)
    labels = model.fit_predict(X)
    return model, labels
