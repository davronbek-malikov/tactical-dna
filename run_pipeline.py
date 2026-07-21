"""
run_pipeline.py — end-to-end player-archetype clustering pipeline.

Run from project root:
    python run_pipeline.py

Steps:
    1. Load raw data (3 leagues, 8 seasons, understat advanced stats)
    2. Feature engineering (per-90 rates, no target — this is unsupervised)
    3. Scale features
    4. Bake-off: KMeans / Agglomerative / GaussianMixture / DBSCAN,
       scored on Silhouette, Davies-Bouldin, Calinski-Harabasz
    5. Fit the chosen final model (KMeans, k=6 — see README for the
       statistical-optimum-vs-interpretability trade-off writeup)
    6. Rule-based archetype naming from cluster centroids
    7. PCA(2) projection for the 2D map (stable transform() for new points,
       unlike t-SNE/UMAP — matters for a live Streamlit demo)
    8. Save all artifacts: model, scaler, pca, labels, clustered dataset
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.clustering import run_bakeoff
from src.labels import label_clusters

RANDOM_STATE = 42
FINAL_K = 6
MODEL_DIR = Path("models")
DATA_OUT = Path("data/processed")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("STEP 1 — Loading raw data")
    df_raw = DataLoader("data/raw").load_all()
    print(f"  {len(df_raw):,} player-season rows across 3 leagues, 8 seasons")

    print("STEP 2 — Feature engineering (per-90 rates, qualified players only)")
    fe = FeatureEngineer()
    df = fe.transform(df_raw)
    feats = FeatureEngineer.CLUSTER_FEATURES
    print(f"  {len(df):,} qualified player-seasons (>=900 min), {len(feats)} features")

    print("STEP 3 — Scaling")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats].values)

    print("STEP 4 — Algorithm bake-off (KMeans / Agglomerative / GMM / DBSCAN)")
    board = run_bakeoff(X, random_state=RANDOM_STATE)
    board.to_csv(MODEL_DIR / "bakeoff_results.csv", index=False)
    print(board.head(10).to_string(index=False))

    print(f"STEP 5 — Fitting final model: KMeans k={FINAL_K}")
    model = KMeans(n_clusters=FINAL_K, n_init=10, random_state=RANDOM_STATE)
    df["cluster"] = model.fit_predict(X)

    print("STEP 6 — Naming archetypes from centroids")
    centroids = df.groupby("cluster")[feats].mean()
    label_map = {int(k): v for k, v in label_clusters(centroids).items()}
    df["archetype"] = df["cluster"].map(label_map)
    print(json.dumps(label_map, indent=2))

    print("STEP 7 — PCA(2) projection for visualization")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    df["pca_x"], df["pca_y"] = coords[:, 0], coords[:, 1]
    print(f"  explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    print("STEP 8 — Saving artifacts")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(model, MODEL_DIR / "kmeans_model.joblib")
    joblib.dump(pca, MODEL_DIR / "pca.joblib")
    with open(MODEL_DIR / "cluster_labels.json", "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    joblib.dump(centroids, MODEL_DIR / "centroids.joblib")

    keep_cols = (
        ["player", "team", "league", "season", "position_group", "cluster", "archetype", "pca_x", "pca_y"]
        + feats
    )
    df[keep_cols].to_csv(DATA_OUT / "players_clustered.csv", index=False)

    print("\nDone. Cluster sizes:")
    print(df["archetype"].value_counts())


if __name__ == "__main__":
    main()
