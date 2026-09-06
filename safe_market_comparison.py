from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from daily_predictions import build_daily_predictions
from market_comparison import compare_totals_25, fetch_market_odds
from pipeline import load_config


def _load_cached_predictions(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    required = {"Date", "HomeTeam", "AwayTeam", "P_HomeWin", "P_Draw", "P_AwayWin"}
    return df if not df.empty and required.issubset(df.columns) else pd.DataFrame()


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "daily_predictions_latest.csv"

    try:
        predictions = build_daily_predictions(cfg)
        if predictions.empty:
            raise RuntimeError("fresh prediction build returned zero fixtures")
        predictions.to_csv(pred_path, index=False)
        print(f"Fresh daily predictions built: {len(predictions)}")
    except (requests.RequestException, RuntimeError, OSError, ValueError) as exc:
        predictions = _load_cached_predictions(pred_path)
        if predictions.empty:
            raise RuntimeError(
                f"Fresh prediction build failed and no valid cached predictions are available: {exc}"
            ) from exc
        print(
            "WARNING: fresh prediction build failed; using the already-published "
            f"prediction snapshot instead. Cause: {exc.__class__.__name__}: {exc}"
        )
        print(f"Cached prediction fixtures used: {len(predictions)}")

    odds = fetch_market_odds(cfg)
    odds.to_csv(out_dir / "market_odds_latest.csv", index=False)
    comparison = compare_totals_25(predictions, odds)
    comparison.to_csv(out_dir / "market_comparison_latest.csv", index=False)

    print(f"Daily fixtures available: {len(predictions)}")
    print(f"Market outcome rows:      {len(odds)}")
    print(f"O/U 2.5 comparisons:     {len(comparison)}")
    if not comparison.empty:
        my_count = int(comparison["MyBookieAvailable"].fillna(False).sum()) if "MyBookieAvailable" in comparison.columns else 0
        print(f"MyBookie comparisons:    {my_count}")


if __name__ == "__main__":
    main()
