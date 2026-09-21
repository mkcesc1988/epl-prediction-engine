from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")
MANUAL_PATH = Path("data/manual_user_picks.csv")
OUT_PATH = DATA_DIR / "benchmark_predictions_latest.csv"


def _model_rows(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    work = rankings.copy()
    status = work.get("DecisionStatus", pd.Series("NO_BET", index=work.index)).astype(str)
    work = work[status.eq("BET_ELIGIBLE")].copy()
    if work.empty:
        return work
    work["BenchmarkSource"] = "MODEL"
    work["BenchmarkProbability"] = pd.to_numeric(work.get("ModelWinProbability"), errors="coerce")
    work["BenchmarkOdds"] = pd.to_numeric(work.get("MyBookieOdds"), errors="coerce")
    work["BenchmarkStakeUnits"] = 1.0
    return work


def _market_rows(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty or "MarketType" not in rankings.columns:
        return pd.DataFrame()
    work = rankings[rankings["MarketType"].astype(str).eq("Moneyline")].copy()
    if work.empty:
        return work
    work["BenchmarkOdds"] = pd.to_numeric(work.get("MyBookieOdds"), errors="coerce")
    work = work[work["BenchmarkOdds"].gt(1.0)].copy()
    if work.empty:
        return work
    keys = ["Date", "HomeTeam", "AwayTeam"]
    idx = work.groupby(keys, dropna=False)["BenchmarkOdds"].idxmin()
    work = work.loc[idx].copy()
    work["BenchmarkSource"] = "MARKET_FAVORITE"
    work["BenchmarkProbability"] = pd.to_numeric(work.get("ConsensusFairProbability"), errors="coerce")
    work["BenchmarkStakeUnits"] = 1.0
    return work


def _manual_rows(path: Path = MANUAL_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    work = pd.read_csv(path)
    required = {"Date", "HomeTeam", "AwayTeam", "MarketType", "Selection", "Odds"}
    missing = required.difference(work.columns)
    if missing:
        raise RuntimeError(f"Manual picks file missing columns: {sorted(missing)}")
    work["BenchmarkSource"] = "USER"
    work["BenchmarkOdds"] = pd.to_numeric(work["Odds"], errors="coerce")
    work["BenchmarkProbability"] = pd.to_numeric(work.get("Probability"), errors="coerce")
    stake = work["StakeUnits"] if "StakeUnits" in work.columns else pd.Series(1.0, index=work.index)
    work["BenchmarkStakeUnits"] = pd.to_numeric(stake, errors="coerce").fillna(1.0)
    return work


def build_benchmark_snapshot(rankings: pd.DataFrame, manual: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = [_model_rows(rankings), _market_rows(rankings)]
    if manual is None:
        manual = _manual_rows()
    elif not manual.empty:
        manual = manual.copy()
        manual["BenchmarkSource"] = "USER"
        manual["BenchmarkOdds"] = pd.to_numeric(manual.get("Odds"), errors="coerce")
        manual["BenchmarkProbability"] = pd.to_numeric(manual.get("Probability"), errors="coerce")
        stake = manual["StakeUnits"] if "StakeUnits" in manual.columns else pd.Series(1.0, index=manual.index)
        manual["BenchmarkStakeUnits"] = pd.to_numeric(stake, errors="coerce").fillna(1.0)
    if manual is not None and not manual.empty:
        frames.append(manual)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    preferred = [
        "BenchmarkSource", "Date", "HomeTeam", "AwayTeam", "MarketType",
        "Selection", "Line", "BenchmarkOdds", "BenchmarkProbability",
        "BenchmarkStakeUnits", "DecisionStatus", "ProbabilityEdge",
        "ConsensusFairProbability", "RawModelWinProbability", "ModelWinProbability",
    ]
    cols = [c for c in preferred if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def main() -> None:
    rankings_path = DATA_DIR / "gameweek_rankings_latest.csv"
    if not rankings_path.exists():
        raise RuntimeError("Run gameweek rankings before benchmark tracker")
    rankings = pd.read_csv(rankings_path)
    out = build_benchmark_snapshot(rankings)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Benchmark rows: {len(out)}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
