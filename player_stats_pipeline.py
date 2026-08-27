"""Reproducible EPL player-stat pipeline for the player-props shadow model.

V0.1 ingests provider-neutral match/player rows and builds leakage-safe rolling
features for shots and shots on target. It deliberately keeps data acquisition
separate, so a provider can be swapped without changing the model layer.

Expected raw CSV columns:
  fixture_id,kickoff_utc,player,team,opponent,minutes,started,shots,shots_on_target

Optional identity columns:
  player_id,team_id,opponent_id

Outputs:
  data/processed/player_match_stats.csv
  data/processed/player_form_features.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {
    "fixture_id", "kickoff_utc", "player", "team", "opponent",
    "minutes", "started", "shots", "shots_on_target",
}
COUNT_COLS = ["shots", "shots_on_target"]
WINDOWS = (5, 10, 20)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Missing required player-stat columns: {missing}")
    out = df.copy()
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True, errors="raise")
    for col in ["minutes", "shots", "shots_on_target"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    out["started"] = out["started"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    if (out["minutes"] < 0).any() or (out["minutes"] > 120).any():
        raise ValueError("minutes must be between 0 and 120")
    if (out[COUNT_COLS] < 0).any().any():
        raise ValueError("shot counts cannot be negative")
    if (out["shots_on_target"] > out["shots"]).any():
        raise ValueError("shots_on_target cannot exceed shots")
    return out.sort_values(["player", "kickoff_utc", "fixture_id"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-match rolling features, shifted one appearance to avoid leakage."""
    out = validate(df)
    group_key = "player_id" if "player_id" in out.columns else "player"
    g = out.groupby(group_key, sort=False, group_keys=False)

    out["prior_appearances"] = g.cumcount()
    out["prior_starts"] = g["started"].transform(lambda s: s.shift(1).fillna(False).cumsum())

    for window in WINDOWS:
        prior_minutes = g["minutes"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        out[f"minutes_last_{window}"] = prior_minutes
        out[f"avg_minutes_last_{window}"] = g["minutes"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        out[f"start_rate_last_{window}"] = g["started"].transform(
            lambda s: s.astype(float).shift(1).rolling(window, min_periods=1).mean()
        )
        for stat in COUNT_COLS:
            prior_count = g[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
            out[f"{stat}_last_{window}"] = prior_count
            out[f"{stat}_per90_last_{window}"] = (90.0 * prior_count / prior_minutes).where(prior_minutes > 0)

    # Season-to-date style expanding rates, also strictly pre-match.
    prior_minutes_all = g["minutes"].transform(lambda s: s.shift(1).expanding(min_periods=1).sum())
    out["minutes_prior_all"] = prior_minutes_all
    for stat in COUNT_COLS:
        prior_stat_all = g[stat].transform(lambda s: s.shift(1).expanding(min_periods=1).sum())
        out[f"{stat}_prior_all"] = prior_stat_all
        out[f"{stat}_per90_prior_all"] = (90.0 * prior_stat_all / prior_minutes_all).where(prior_minutes_all > 0)

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/player_match_stats.csv")
    p.add_argument("--clean-output", default="data/processed/player_match_stats.csv")
    p.add_argument("--feature-output", default="data/processed/player_form_features.csv")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists() or inp.stat().st_size == 0:
        print(f"No raw player-stat file at {inp}. Pipeline exits cleanly.")
        return

    raw = pd.read_csv(inp)
    if raw.empty:
        print("Raw player-stat file contains no rows. Pipeline exits cleanly.")
        return

    clean = validate(raw)
    features = build_features(clean)
    clean_path = Path(args.clean_output)
    feature_path = Path(args.feature_output)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False)
    features.to_csv(feature_path, index=False)
    print(f"Player pipeline complete: {len(clean)} match-player rows, {clean['player'].nunique()} players.")


if __name__ == "__main__":
    main()
