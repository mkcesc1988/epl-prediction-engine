"""EPL Player Props V0.1, shadow-only research engine.

This module intentionally does NOT place or recommend real-money bets.
It converts externally supplied player projections and sportsbook lines into
probabilities, fair odds, EV, and a paper-tracking decision.

Initial supported markets:
  * shots
  * shots_on_target

Input CSV columns:
fixture_id,kickoff_utc,player,team,opponent,market,line,decimal_odds,
expected_per90,expected_minutes

Optional columns:
bookmaker,projection_source,starter_probability
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd
from scipy.stats import poisson

SUPPORTED_MARKETS = {"shots", "shots_on_target"}
DEFAULT_MIN_EDGE = 0.05
DEFAULT_MIN_EV = 0.05
DEFAULT_MIN_MINUTES = 55.0


def over_probability(lam: float, line: float) -> float:
    """Probability a count finishes strictly over a standard x.5 line."""
    if line < 0 or abs((line * 2) - round(line * 2)) > 1e-9 or int(round(line * 2)) % 2 == 0:
        raise ValueError("V0.1 supports half-point count lines only, e.g. 2.5")
    threshold = math.floor(line)
    return float(poisson.sf(threshold, lam))


def under_probability(lam: float, line: float) -> float:
    return 1.0 - over_probability(lam, line)


def build_shadow_board(df: pd.DataFrame, min_edge: float = DEFAULT_MIN_EDGE,
                       min_ev: float = DEFAULT_MIN_EV,
                       min_minutes: float = DEFAULT_MIN_MINUTES) -> pd.DataFrame:
    required = {
        "fixture_id", "kickoff_utc", "player", "team", "opponent", "market",
        "line", "decimal_odds", "expected_per90", "expected_minutes"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["market"] = out["market"].astype(str).str.lower().str.strip()
    bad = sorted(set(out.loc[~out["market"].isin(SUPPORTED_MARKETS), "market"]))
    if bad:
        raise ValueError(f"Unsupported markets in V0.1: {bad}")

    for c in ["line", "decimal_odds", "expected_per90", "expected_minutes"]:
        out[c] = pd.to_numeric(out[c], errors="raise")
    if (out["decimal_odds"] <= 1).any():
        raise ValueError("decimal_odds must be > 1.0")
    if (out["expected_minutes"] < 0).any() or (out["expected_minutes"] > 120).any():
        raise ValueError("expected_minutes must be between 0 and 120")

    if "starter_probability" not in out.columns:
        out["starter_probability"] = 1.0
    out["starter_probability"] = pd.to_numeric(out["starter_probability"], errors="raise").clip(0, 1)

    out["lambda_count"] = (
        out["expected_per90"] * out["expected_minutes"] / 90.0 * out["starter_probability"]
    )
    out["model_probability"] = [
        over_probability(lam, line) for lam, line in zip(out["lambda_count"], out["line"])
    ]
    out["model_fair_odds"] = 1.0 / out["model_probability"].clip(lower=1e-9)
    out["book_implied_probability"] = 1.0 / out["decimal_odds"]
    out["probability_edge"] = out["model_probability"] - out["book_implied_probability"]
    out["expected_return_per_unit"] = out["model_probability"] * out["decimal_odds"] - 1.0

    # Shadow qualification is deliberately conservative while the prop model is unvalidated.
    out["shadow_qualified"] = (
        (out["expected_minutes"] >= min_minutes)
        & (out["starter_probability"] >= 0.80)
        & (out["probability_edge"] >= min_edge)
        & (out["expected_return_per_unit"] >= min_ev)
    )
    out["prop_model_version"] = "PLAYER_PROPS_V0.1_SHADOW"
    out["real_money_eligible"] = False
    out["status"] = out["shadow_qualified"].map({True: "PAPER_TRACK", False: "PASS"})

    sort_cols = ["shadow_qualified", "expected_return_per_unit", "probability_edge"]
    return out.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/player_props_input.csv")
    p.add_argument("--output", default="data/processed/player_props_shadow_latest.csv")
    p.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    p.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV)
    p.add_argument("--min-minutes", type=float, default=DEFAULT_MIN_MINUTES)
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists() or inp.stat().st_size == 0:
        print(f"No player-prop input available at {inp}. Shadow engine exits cleanly.")
        return
    df = pd.read_csv(inp)
    if df.empty:
        print("Player-prop input has headers but no rows. Shadow engine exits cleanly.")
        return

    board = build_shadow_board(df, args.min_edge, args.min_ev, args.min_minutes)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out, index=False)
    print(f"Wrote {len(board)} player props to {out}, {int(board['shadow_qualified'].sum())} paper-qualified.")


if __name__ == "__main__":
    main()
