"""Conservative, explicitly hypothetical lineup scenarios for V1.3.

Overrides are sensitivity assumptions, not estimated player effects.
Never overwrite baseline predictions or claim confirmed starting elevens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model_v11 import _score_matrix
from gameweek_rankings import _moneyline_prob, _spread_prob, _totals_prob

DATA_DIR = Path("data/processed")
INPUT_PATH = Path("data/manual_scenarios_v13.csv")
OUTPUT_PATH = DATA_DIR / "scenario_predictions_v13_latest.csv"


def _probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    h = float(np.tril(matrix, k=-1).sum())
    d = float(np.trace(matrix))
    a = float(np.triu(matrix, k=1).sum())
    return h, d, a


def build_scenarios(predictions: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "HomeTeam", "AwayTeam", "HomeLambdaMultiplier",
                "AwayLambdaMultiplier", "AssumptionNote"}
    if predictions.empty or overrides.empty or not required.issubset(overrides.columns):
        return pd.DataFrame()
    output = []
    for _, s in overrides.iterrows():
        matched = predictions[
            (predictions["Date"].astype(str) == str(s["Date"]))
            & (predictions["HomeTeam"].astype(str) == str(s["HomeTeam"]))
            & (predictions["AwayTeam"].astype(str) == str(s["AwayTeam"]))
        ]
        if len(matched) != 1:
            continue
        r = matched.iloc[0]
        hm, am = float(s["HomeLambdaMultiplier"]), float(s["AwayLambdaMultiplier"])
        if not np.isfinite(hm) or not np.isfinite(am) or not (0.85 <= hm <= 1.15 and 0.85 <= am <= 1.15):
            raise ValueError("All scenario xG multipliers must be within +/-15%")
        lh, la = float(r["Lambda_Home_xG"]), float(r["Lambda_Away_xG"])
        rho = float(r["DixonColes_Rho"])
        raw = _score_matrix(lh, la, rho, 10)
        scen = _score_matrix(lh * hm, la * am, rho, 10)
        bh, bd, ba = _probabilities(raw)
        sh, sd, sa = _probabilities(scen)
        output.append({
            "Date": str(s["Date"]), "HomeTeam": str(s["HomeTeam"]), "AwayTeam": str(s["AwayTeam"]),
            "BaselineHomeLambda": lh, "BaselineAwayLambda": la,
            "ScenarioHomeLambda": lh * hm, "ScenarioAwayLambda": la * am,
            "BaseHomeWin": bh, "BaseDraw": bd, "BaseAwayWin": ba,
            "ScenarioHomeWin": sh, "ScenarioDraw": sd, "ScenarioAwayWin": sa,
            "AssumptionNote": str(s["AssumptionNote"]),
            "EvidenceURL": str(s.get("EvidenceURL", "")),
            "ScenarioStatus": "UNVALIDATED_HYPOTHESIS_ONLY",
        })
    return pd.DataFrame(output)


def scenario_probability(row: pd.Series, scenario: pd.Series) -> tuple[float, float]:
    """Compute raw market probability for the specified scenario.

    Supports 1X2, totals and spreads. Full and quarter lines are handled by
    callers' existing settlement logic; the V1.3 decision gate only applies
    probability stress tests to markets with zero push probability.
    """
    matrix = _score_matrix(
        float(scenario["ScenarioHomeLambda"]),
        float(scenario["ScenarioAwayLambda"]),
        -0.075, 10,
    )
    # Use source rho when available to ensure parity with baseline.
    rho = row.get("DixonColes_Rho", -0.075)
    if pd.notna(rho):
        matrix = _score_matrix(float(scenario["ScenarioHomeLambda"]),
                               float(scenario["ScenarioAwayLambda"]), float(rho), 10)
    market = str(row.get("MarketType"))
    selection = str(row.get("Selection"))
    if market == "Moneyline":
        h, d, a = _probabilities(matrix)
        home, away = str(row["HomeTeam"]), str(row["AwayTeam"])
        if selection == home:
            return h, 0.0
        if selection == away:
            return a, 0.0
        if selection.lower() == "draw":
            return d, 0.0
    elif market == "Spread":
        return _spread_prob(matrix, str(row["HomeTeam"]), str(row["AwayTeam"]),
                            selection.split(" +")[0].split(" -")[0], float(row["Line"]))
    elif market == "Total":
        return _totals_prob(matrix, selection.split(" ")[0], float(row["Line"]))
    return float("nan"), float("nan")


def main() -> None:
    if not INPUT_PATH.exists():
        print("No scenario overrides supplied; no V1.3 scenario analysis")
        return
    pred = pd.read_csv(DATA_DIR / "daily_predictions_latest.csv")
    scenarios = build_scenarios(pred, pd.read_csv(INPUT_PATH))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(OUTPUT_PATH, index=False)
    if not scenarios.empty:
        display = ["HomeTeam", "AwayTeam", "ScenarioHomeLambda", "ScenarioAwayLambda",
                   "ScenarioHomeWin", "ScenarioDraw", "ScenarioAwayWin", "ScenarioStatus"]
        print(scenarios[display].to_string(index=False))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
