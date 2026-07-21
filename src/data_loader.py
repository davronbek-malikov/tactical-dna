"""
DataLoader — loads raw per-season league files and returns one clean
player-season DataFrame. No salary join here: this is an unsupervised
project, there is no target, so nothing downstream can leak.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

LEAGUES = [
    ("Bundesliga", "bundesliga", "Bundesliga"),
    ("Laliga", "laliga", "La Liga"),
    ("Serie A", "serie_a", "Serie A"),
]
SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"]


class DataLoader:
    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    def _load_league(self, folder: str, prefix: str, name: str) -> pd.DataFrame:
        frames = []
        for season in SEASONS:
            path = self.raw_dir / folder / f"metadata_{prefix}_{season}.xls"
            if not path.exists():
                continue
            df = pd.read_csv(str(path))
            df["league"] = name
            df["season"] = season
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_all(self) -> pd.DataFrame:
        parts = [self._load_league(*args) for args in LEAGUES]
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        df.rename(columns={"player_name": "player", "team_title": "team"}, inplace=True)
        return df
