"""
Soccer Player Archetype Explorer — Streamlit demo.

Two modes:
  1. Pick a real player-season from the Big-5-league dataset (2014/15-2021/22)
  2. Enter a hypothetical player's raw season stats

Both are assigned to one of 6 data-driven archetypes (KMeans, k=6 — see
README for why this k was chosen over the statistically "optimal" k=3),
shown against their cluster's centroid on a radar chart, placed on the
PCA map, and matched to the 5 most similar real players.
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.feature_engineer import FeatureEngineer
from src.similarity import find_similar_players

FEATS = FeatureEngineer.CLUSTER_FEATURES
MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/players_clustered.csv")

st.set_page_config(page_title="Soccer Player Archetype Explorer", page_icon="⚽", layout="wide")


@st.cache_resource
def load_artifacts():
    # Safe: loading our own sklearn/joblib artifacts written by run_pipeline.py
    # in this repo — never loads files from an untrusted source.
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    model = joblib.load(MODEL_DIR / "kmeans_model.joblib")
    pca = joblib.load(MODEL_DIR / "pca.joblib")
    with open(MODEL_DIR / "cluster_labels.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    df = pd.read_csv(DATA_PATH, dtype={"season": str})
    X = scaler.transform(df[FEATS].values)
    return scaler, model, pca, label_map, df, X


scaler, model, pca, label_map, df, X_all = load_artifacts()
mins = df[FEATS].min()
maxs = df[FEATS].max()


def normalize_for_radar(vec: dict) -> list[float]:
    return [
        float(np.clip((vec[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9), 0, 1))
        for f in FEATS
    ]


def radar_chart(player_vec: dict, centroid_vec: dict, player_name: str, archetype: str):
    categories = FEATS + [FEATS[0]]
    r_player = normalize_for_radar(player_vec)
    r_player.append(r_player[0])
    r_centroid = normalize_for_radar(centroid_vec)
    r_centroid.append(r_centroid[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r_centroid, theta=categories, fill="toself",
                                   name=f"{archetype} (cluster avg)", opacity=0.4))
    fig.add_trace(go.Scatterpolar(r=r_player, theta=categories, fill="toself",
                                   name=player_name, opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       showlegend=True, height=480, margin=dict(t=30, b=30))
    return fig


def pca_map(highlight_xy=None, highlight_name="Your player"):
    fig = go.Figure()
    for cid, name in label_map.items():
        sub = df[df["cluster"] == cid]
        fig.add_trace(go.Scattergl(
            x=sub["pca_x"], y=sub["pca_y"], mode="markers", name=name,
            marker=dict(size=5, opacity=0.5),
            text=sub["player"] + " (" + sub["season"] + ")",
            hoverinfo="text",
        ))
    if highlight_xy is not None:
        fig.add_trace(go.Scatter(
            x=[highlight_xy[0]], y=[highlight_xy[1]], mode="markers",
            marker=dict(size=16, color="black", symbol="star"), name=highlight_name,
        ))
    fig.update_layout(height=520, xaxis_title="PCA 1", yaxis_title="PCA 2",
                       legend=dict(orientation="h", yanchor="bottom", y=-0.3))
    return fig


st.title("⚽ Soccer Player Archetype Explorer")
st.caption(
    "Unsupervised clustering (KMeans, k=6) over 6,314 qualified player-seasons "
    "(≥900 minutes) from Bundesliga, La Liga & Serie A, 2014/15–2021/22 — "
    "same dataset family as the Salary Predictor project, this time with no target label."
)

tab1, tab2 = st.tabs(["🔍 Explore a real player", "🧪 Try a custom player"])

with tab1:
    df["label"] = df["player"] + " — " + df["team"] + " (" + df["season"] + ", " + df["league"] + ")"
    choice = st.selectbox("Pick a player-season", df["label"].sort_values().unique(), index=0)
    row = df[df["label"] == choice].iloc[0]
    cluster = int(row["cluster"])
    archetype = row["archetype"]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Archetype", archetype)
        st.write(f"**Position group:** {row['position_group']}  ·  **Cluster:** {cluster}")
        player_vec = {f: row[f] for f in FEATS}
        centroid_vec = df[df["cluster"] == cluster][FEATS].mean().to_dict()
        st.plotly_chart(radar_chart(player_vec, centroid_vec, row["player"], archetype), use_container_width=True)

    with c2:
        st.plotly_chart(pca_map((row["pca_x"], row["pca_y"]), row["player"]), use_container_width=True)

    st.subheader(f"Players most similar to {row['player']} ({row['season']})")
    query_vec = X_all[df.index[df["label"] == choice][0]]
    similar = find_similar_players(df, X_all, query_vec, cluster, FEATS, top_n=6)
    similar = similar[similar["player"] != row["player"]].head(5)
    st.dataframe(
        similar[["player", "team", "season", "similarity"]].style.format({"similarity": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

with tab2:
    st.write("Enter a season's raw totals — the app computes per-90 rates the same way the model was trained.")
    c1, c2, c3 = st.columns(3)
    with c1:
        games = st.number_input("Games", 1, 60, 30)
        minutes = st.number_input("Minutes played", 90, 5400, 2500)
        goals = st.number_input("Goals", 0, 60, 10)
        assists = st.number_input("Assists", 0, 40, 6)
    with c2:
        xG = st.number_input("xG", 0.0, 50.0, 9.0)
        xA = st.number_input("xA", 0.0, 30.0, 5.0)
        shots = st.number_input("Shots", 0, 300, 60)
        key_passes = st.number_input("Key passes", 0, 200, 40)
    with c3:
        xGChain = st.number_input("xGChain", 0.0, 40.0, 12.0)
        xGBuildup = st.number_input("xGBuildup", 0.0, 30.0, 6.0)
        yellow = st.number_input("Yellow cards", 0, 20, 3)
        red = st.number_input("Red cards", 0, 5, 0)

    if st.button("Classify this player", type="primary"):
        feat_vec = FeatureEngineer.compute_from_raw(
            minutes, games, goals, assists, xG, xA, shots, key_passes,
            xGChain, xGBuildup, yellow, red,
        )
        x_row = np.array([[feat_vec[f] for f in FEATS]])
        x_scaled = scaler.transform(x_row)
        pred_cluster = int(model.predict(x_scaled)[0])
        pred_archetype = label_map[pred_cluster]
        pred_xy = pca.transform(x_scaled)[0]

        st.success(f"**Predicted archetype: {pred_archetype}**")
        c1, c2 = st.columns([1, 1])
        with c1:
            centroid_vec = df[df["cluster"] == pred_cluster][FEATS].mean().to_dict()
            st.plotly_chart(radar_chart(feat_vec, centroid_vec, "Your player", pred_archetype), use_container_width=True)
        with c2:
            st.plotly_chart(pca_map(pred_xy, "Your player"), use_container_width=True)

        st.subheader("Most similar real players")
        similar = find_similar_players(df, X_all, x_scaled[0], pred_cluster, FEATS, top_n=5)
        st.dataframe(
            similar[["player", "team", "season", "similarity"]].style.format({"similarity": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

st.divider()
st.caption(
    "Built by Davronbek Malikov · [GitHub](https://github.com/davronbek-malikov) · "
    "Companion project to the [Soccer Salary Predictor](https://soccer-xg-salary-prediction.onrender.com)."
)
