# 🧬 TacticalDNA — Unsupervised Player-Archetype Discovery

TacticalDNA reads a player the way genetics reads a genome — no coach's
opinion, no scouting report, no salary figure. Feed it 6,300+ player-seasons
from Europe's top 5 leagues (Bundesliga, La Liga, Serie A · 2014/15–2021/22)
and it discovers six distinct playing identities entirely on its own, purely
from what happened on the pitch — then a Streamlit app lets you classify any
player, real or hypothetical, and find their closest statistical twins.

> **The interesting part:** during testing, a completely made-up stat line —
> "25 goals, 15 assists, 2,500 minutes," no name attached — landed the model
> in the *Elite Complete Forward* cluster, matched to prime **Messi's 2019/20
> season at 98% cosine similarity**. Nobody told the model who Messi is. It
> found that identity purely from the shape of the numbers.

**Companion project to [Soccer Salary Predictor](https://github.com/davronbek-malikov/soccer-xG-salary_prediction)** — same dataset family, this time solving an unsupervised problem (clustering) instead of supervised regression, using the same understat-derived Big-5-league data.

🔗 **Live demo:** https://soccer-player-clustering.onrender.com
🔗 **Salary Predictor (sister project):** https://soccer-xg-salary-prediction.onrender.com

---

## Problem

Scouting and recruitment teams don't just want "how much is this player
worth" (the salary project) — they want "what *kind* of player is this,
and who else plays like them." That's a clustering problem: group players
by playing style using only their on-pitch output, with no salary, name,
or reputation involved in the grouping itself.

## Data & feature engineering

- **Source:** understat-derived match/season aggregates (same raw files as
  the salary project) — goals, xG, assists, xA, shots, key passes, cards,
  xGChain, xGBuildup, position, per player-season.
- **Qualification filter:** players with **< 900 minutes** in a season are
  dropped. Per-90 rates on a handful of appearances are noise (one goal in
  180 minutes is not a signal) — this is the same "qualified players"
  threshold analytics sites apply.
- **Per-90 normalization:** every counting stat is converted to a per-90-minutes
  rate so a impact substitute is comparable to a player who started every game.
- **Engineered features (13, all rate/ratio — no raw totals, so playing time
  itself never drives the clustering):**

  | Feature | What it captures |
  |---|---|
  | `goals_p90`, `assists_p90` | direct output |
  | `xG_p90`, `xA_p90` | quality of chances taken/created |
  | `shots_p90`, `key_passes_p90` | attacking involvement volume |
  | `xGChain_p90`, `xGBuildup_p90` | involvement in the buildup, not just the end product |
  | `shot_conversion` | finishing efficiency (goals / shots) |
  | `finishing_over_xG`, `creation_over_xA` | over/underperformance vs. expected |
  | `discipline_p90` | cards per 90 |
  | `involvement_index` | composite build-up-play score |

  Goalkeepers are excluded (outfield-only feature set). No salary, age, or
  team is used anywhere in the clustering — those would leak identity/market
  signal into what is supposed to be a pure style grouping.

## Algorithm bake-off

Four families tested, each scored on **three independent internal-validation
metrics** (no ground-truth labels exist for "playing style", so all scoring
is unsupervised): Silhouette (higher = better separated), Davies-Bouldin
(lower = better), Calinski-Harabasz (higher = better). Composite rank = mean
rank across all three.

| Algorithm | Params | Silhouette | Davies-Bouldin | Calinski-Harabasz | Composite rank |
|---|---|---|---|---|---|
| **KMeans** | **k=3** | **0.285** | 1.783 | **1786.5** | **1st** |
| KMeans | k=5 | 0.217 | 1.666 | 1375.0 | 3rd |
| KMeans | k=4 | 0.214 | 1.738 | 1546.0 | 3rd |
| **KMeans** | **k=6** | 0.179 | 1.638 | 1270.6 | 4th |
| KMeans | k=7 | 0.181 | 1.664 | 1184.9 | 5th |
| KMeans | k=8 | 0.174 | 1.595 | 1117.1 | 6th |
| Agglomerative (Ward) | k=3–8 | 0.08–0.17 | 1.88–2.03 | 850–1384 | 7th–11th |
| Gaussian Mixture | k=3–8 | 0.01–0.08 | 2.16–3.22 | 547–1187 | 11th–18th |
| DBSCAN | eps grid × min_samples∈{10,15,20} | — | — | — | *degenerate* |

**DBSCAN** collapsed to one dense cluster (95–98% of points) plus a noise
tail at every eps/min_samples tested — expected behaviour, since continuous
per-90 performance stats form a smooth density gradient rather than
density-separated islands. It's a legitimate empirical result, not a bug:
density-based clustering isn't the right tool for this feature space.

**KMeans wins outright** — best on all three metrics, zero noise points,
stable across `n_init=10` restarts.

### Choosing k: statistical optimum vs. product usefulness

The metrics are unambiguous that **k=3** is the best-separated partition —
but three clusters on this feature set turn out to mostly separate
"low / medium / high overall attacking involvement," which is only a small
step beyond what position alone already tells you. For a scouting/style
tool, that's not a useful product.

**k=6** is the deployed operating point.** It's still a strong KMeans fit
(2nd-best Davies-Bouldin of the whole grid, positive silhouette, zero noise)
and — critically — it produces six clusters that separate cleanly along
recognisable soccer roles when you look at who's actually in them (see
below). This is a real trade-off industry teams make constantly: the
statistically "optimal" k is a starting point, not a mandate — the
deployed k is chosen with the downstream use case in the room.

## The six archetypes

Cluster naming is **rule-based on centroid ranks** (`src/labels.py`), not
hand-picked — the labels below are what the rules produced, validated
against who actually landed in each group:

| Archetype | Signature | Example players (real seasons in the data) |
|---|---|---|
| **Elite Complete Forward** | Highest goals+assists *and* highest overall involvement | Messi (multiple seasons), Ronaldo, Muriel |
| **Clinical Poacher** | Highest goals & xG per 90 among high-volume shooters | Lewandowski (multiple seasons), Alcácer, Muriel |
| **Deep-Lying Playmaker** | Highest build-up involvement, modest direct output | Kimmich, Rakitic, Guerreiro |
| **Efficient Impact Player** | Low touches, highest shot-conversion rate | Pobega, Hoppe, Sandro Ramírez |
| **Volume Forward** | High shot/chance volume, moderate efficiency | Diego Costa, Maxi López, Ibisevic |
| **Defensive Anchor** | Lowest attacking involvement, highest card rate | mostly CBs / holding mids |

## Similarity engine

Given any player (real or hand-entered), cosine similarity is computed in
the scaled 13-feature space **restricted to their predicted cluster** —
comparing a Poacher to other Poachers, not to Defensive Anchors. A
hypothetical "25 goals / 15 assists in 2500 minutes" input correctly lands
in *Elite Complete Forward* with Messi's 2019/20 season as the closest
match (cosine similarity 0.98).

## Why PCA, not t-SNE/UMAP, for the 2D map

t-SNE and UMAP produce prettier separated blobs but don't have a cheap,
stable `.transform()` for a brand-new point — every new player would need
the whole embedding recomputed. That's fine for a static notebook, not for
a live app where a user's hand-entered stats need to be placed on the map
in milliseconds. PCA(2) captures 59.2% of variance and transforms new points
in O(1) — the right trade-off for a deployed demo.

## Streamlit demo

Two modes:
1. **Explore a real player** — pick any of 6,300+ player-seasons, see their
   archetype, a radar chart vs. their cluster's centroid, their spot on the
   PCA map, and their 5 closest statistical twins.
2. **Try a custom player** — enter season totals (goals, assists, xG, xA,
   shots, key passes, minutes, cards) for a hypothetical player and get the
   same breakdown.

## Project structure

```
tactical-dna/
├── app.py                       # Streamlit demo (entry point for Render)
├── run_pipeline.py              # end-to-end: load → engineer → bake-off → fit → save
├── src/
│   ├── data_loader.py           # raw multi-league season loader
│   ├── feature_engineer.py      # per-90 feature engineering (+ single-row variant for the app)
│   ├── clustering.py            # algorithm bake-off (KMeans/Agglomerative/DBSCAN/GMM)
│   ├── labels.py                # rule-based archetype naming from centroid ranks
│   └── similarity.py            # cosine-similarity nearest neighbours within a cluster
├── data/
│   ├── raw/                     # per-season league files (understat export)
│   └── processed/players_clustered.csv
├── models/                      # scaler, kmeans model, pca, cluster_labels.json, bake-off results
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python run_pipeline.py     # rebuilds all model artifacts from raw data
streamlit run app.py
```

## Stack

Python · scikit-learn (KMeans, Agglomerative, DBSCAN, Gaussian Mixture, PCA)
· pandas/numpy · Plotly · Streamlit · Render · UptimeRobot

---

Built by [Davronbek Malikov](https://github.com/davronbek-malikov) ·
[Portfolio](https://davronbek-portfolio.vercel.app) ·
[Soccer Salary Predictor](https://github.com/davronbek-malikov/soccer-xG-salary_prediction) (sister project)
