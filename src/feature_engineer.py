"""
FeatureEngineer — turns raw season totals into per-90 rate stats, the
standard unit in soccer analytics (removes minutes-played bias so a
sub who played 900 minutes is comparable to a starter who played 3000).

MIN_MINUTES qualifies players — below this, per-90 rates are noisy
(a single goal in 180 minutes distorts goals_p90 wildly), so they are
dropped, the same way analytics sites apply a "qualified players" filter.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

EPS = 1e-6
MIN_MINUTES = 900  # ~10 full matches — standard qualification threshold

NUMERIC_RAW = [
    "games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes",
    "yellow_cards", "red_cards", "npg", "npxG", "xGChain", "xGBuildup",
]

POSITION_GROUPS = {
    "GK": "GK",
    "D": "DEF", "DC": "DEF", "DL": "DEF", "DR": "DEF", "DMC": "DEF",
    "DM": "MID", "M": "MID", "MC": "MID", "ML": "MID", "MR": "MID",
    "AM": "MID", "AMC": "MID", "AML": "MID", "AMR": "MID",
    "F": "FWD", "FW": "FWD", "S": "FWD",
}


def _simplify_position(pos: str) -> str:
    if not isinstance(pos, str):
        return "UNK"
    first = pos.split(" ")[0].strip()
    for key, group in POSITION_GROUPS.items():
        if first.upper().startswith(key):
            return group
    return "UNK"


class FeatureEngineer:
    """Stateless — every feature is a deterministic function of the row."""

    CLUSTER_FEATURES = [
        "goals_p90", "assists_p90", "xG_p90", "xA_p90",
        "shots_p90", "key_passes_p90", "xGChain_p90", "xGBuildup_p90",
        "shot_conversion", "finishing_over_xG", "creation_over_xA",
        "discipline_p90", "involvement_index",
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in NUMERIC_RAW:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["time"] >= MIN_MINUTES].reset_index(drop=True)

        p90 = 90 / (df["time"] + EPS)
        df["goals_p90"] = df["goals"] * p90
        df["assists_p90"] = df["assists"] * p90
        df["xG_p90"] = df["xG"] * p90
        df["xA_p90"] = df["xA"] * p90
        df["shots_p90"] = df["shots"] * p90
        df["key_passes_p90"] = df["key_passes"] * p90
        df["xGChain_p90"] = df["xGChain"] * p90
        df["xGBuildup_p90"] = df["xGBuildup"] * p90

        df["shot_conversion"] = df["goals"] / (df["shots"] + EPS)
        df["finishing_over_xG"] = df["goals_p90"] - df["xG_p90"]
        df["creation_over_xA"] = df["assists_p90"] - df["xA_p90"]
        df["discipline_p90"] = (df["yellow_cards"] + 2 * df["red_cards"]) * p90
        df["involvement_index"] = (
            df["xGChain_p90"] + df["xGBuildup_p90"]
        ) / 2

        df["position_group"] = df["position"].apply(_simplify_position)
        df = df[df["position_group"] != "GK"].reset_index(drop=True)

        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=self.CLUSTER_FEATURES
        ).reset_index(drop=True)

        return df

    @staticmethod
    def compute_from_raw(
        minutes: float, games: float, goals: float, assists: float,
        xG: float, xA: float, shots: float, key_passes: float,
        xGChain: float, xGBuildup: float,
        yellow_cards: float = 0.0, red_cards: float = 0.0,
    ) -> dict:
        """Same per-90 formulas as transform(), for a single hand-entered
        player (used by the Streamlit 'custom stats' input mode)."""
        p90 = 90 / (minutes + EPS)
        goals_p90 = goals * p90
        assists_p90 = assists * p90
        xG_p90 = xG * p90
        xA_p90 = xA * p90
        return {
            "goals_p90": goals_p90,
            "assists_p90": assists_p90,
            "xG_p90": xG_p90,
            "xA_p90": xA_p90,
            "shots_p90": shots * p90,
            "key_passes_p90": key_passes * p90,
            "xGChain_p90": xGChain * p90,
            "xGBuildup_p90": xGBuildup * p90,
            "shot_conversion": goals / (shots + EPS),
            "finishing_over_xG": goals_p90 - xG_p90,
            "creation_over_xA": assists_p90 - xA_p90,
            "discipline_p90": (yellow_cards + 2 * red_cards) * p90,
            "involvement_index": ((xGChain * p90) + (xGBuildup * p90)) / 2,
        }
