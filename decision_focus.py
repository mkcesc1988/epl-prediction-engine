from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
OUT_PATH = DATA_DIR / "decision_focus_latest.csv"


def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _norm(name: object) -> str:
    value = str(name).strip()
    mapping = {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Tottenham Hotspur": "Spurs",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton and Hove Albion": "Brighton",
        "Nottingham Forest": "Nott'm Forest",
        "Leeds United": "Leeds",
    }
    return mapping.get(value, value)


def _parse_score(score: object) -> tuple[int, int] | None:
    try:
        left, right = str(score).strip().split("-")
        return int(left), int(right)
    except Exception:
        return None


def _modal_support(row: pd.Series) -> tuple[float, str]:
    score = _parse_score(row.get("MostLikelyScore"))
    if score is None:
        return 0.5, "Modal score unavailable"
    h, a = score
    market = str(row.get("MarketType", ""))
    selection = str(row.get("Selection", "")).strip()
    line = pd.to_numeric(row.get("Line"), errors="coerce")
    home = _norm(row.get("HomeTeam"))
    away = _norm(row.get("AwayTeam"))

    if market == "Moneyline":
        if h > a:
            modal_pick = home
        elif a > h:
            modal_pick = away
        else:
            modal_pick = "Draw"
        support = 1.0 if selection.lower() == modal_pick.lower() else 0.0
        return support, f"Modal score {h}-{a} implies {modal_pick}"

    if market == "Total" and pd.notna(line):
        total = h + a
        if selection.lower().startswith("over"):
            if total > float(line):
                return 1.0, f"Modal score {h}-{a} supports over {float(line):g}"
            if abs(total - float(line)) < 1e-9:
                return 0.5, f"Modal score {h}-{a} lands on total line"
            return 0.0, f"Modal score {h}-{a} opposes over {float(line):g}"
        if selection.lower().startswith("under"):
            if total < float(line):
                return 1.0, f"Modal score {h}-{a} supports under {float(line):g}"
            if abs(total - float(line)) < 1e-9:
                return 0.5, f"Modal score {h}-{a} lands on total line"
            return 0.0, f"Modal score {h}-{a} opposes under {float(line):g}"

    if market == "Spread" and pd.notna(line):
        selected_home = selection.lower().startswith(home.lower())
        selected_away = selection.lower().startswith(away.lower())
        margin = (h - a) if selected_home else (a - h) if selected_away else None
        if margin is not None:
            adjusted = margin + float(line)
            if adjusted > 0:
                return 1.0, f"Modal score {h}-{a} covers the spread"
            if abs(adjusted) < 1e-9:
                return 0.5, f"Modal score {h}-{a} pushes the spread"
            return 0.0, f"Modal score {h}-{a} does not cover the spread"

    return 0.5, f"Modal score {h}-{a} is neutral for this market"


def _agreement_score(value: object) -> float:
    label = str(value)
    return {"STRONG_AGREEMENT": 1.0, "PARTIAL_AGREEMENT": 0.65, "DISAGREEMENT": 0.20}.get(label, 0.5)


def _validation_score(value: object) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.5


def _tier(score: float, ev: float, agreement: str, modal: float) -> str:
    if ev <= 0:
        return "PASS"
    if score >= 78 and agreement == "STRONG_AGREEMENT" and modal >= 0.5:
        return "STRONG_FOCUS"
    if score >= 66:
        return "REVIEW"
    return "LOW_PRIORITY"


def main() -> None:
    rankings = _read("gameweek_rankings_latest.csv")
    adjusted = _read("pre_kickoff_adjusted_latest.csv")

    if rankings.empty:
        pd.DataFrame().to_csv(OUT_PATH, index=False)
        print("Decision focus rows: 0")
        return

    adj_map: dict[tuple[str, str, str], pd.Series] = {}
    if not adjusted.empty:
        for _, r in adjusted.iterrows():
            adj_map[(str(r.get("Date")), _norm(r.get("HomeTeam")), _norm(r.get("AwayTeam")))] = r

    rows = []
    for _, r in rankings.iterrows():
        key = (str(r.get("Date")), _norm(r.get("HomeTeam")), _norm(r.get("AwayTeam")))
        a = adj_map.get(key)
        agreement = str(a.get("Agreement")) if a is not None else "UNKNOWN"
        agreement_component = _agreement_score(agreement)
        modal_component, modal_note = _modal_support(r)
        overall = pd.to_numeric(r.get("OverallRankScore"), errors="coerce")
        overall_component = float(np.clip((0.0 if pd.isna(overall) else float(overall)) / 100.0, 0.0, 1.0))
        validation_component = _validation_score(r.get("ValidationFactor"))
        ev = pd.to_numeric(r.get("ExpectedReturnPerUnit"), errors="coerce")
        ev_value = 0.0 if pd.isna(ev) else float(ev)

        focus_score = 100.0 * (
            0.50 * overall_component
            + 0.20 * agreement_component
            + 0.15 * modal_component
            + 0.15 * validation_component
        )
        tier = _tier(focus_score, ev_value, agreement, modal_component)

        adjusted_pick = a.get("AdjustedPick") if a is not None else pd.NA
        adjusted_pick_p = a.get("AdjustedPickProbability") if a is not None else pd.NA
        market_status = a.get("MarketStatus") if a is not None else pd.NA

        reasons = []
        if agreement == "STRONG_AGREEMENT":
            reasons.append("V1.2, V2 context and adjusted direction agree")
        elif agreement == "PARTIAL_AGREEMENT":
            reasons.append("Partial model agreement")
        elif agreement == "DISAGREEMENT":
            reasons.append("Model disagreement lowers confidence")
        reasons.append(modal_note)
        if ev_value > 0:
            reasons.append(f"Positive modeled EV {ev_value * 100:.1f}%")
        else:
            reasons.append("No positive modeled EV")

        row = r.to_dict()
        row.update({
            "DecisionFocusScore": focus_score,
            "DecisionTier": tier,
            "Agreement": agreement,
            "ModalSupport": modal_component,
            "ModalSupportNote": modal_note,
            "AdjustedPick": adjusted_pick,
            "AdjustedPickProbability": adjusted_pick_p,
            "AdjustedMarketStatus": market_status,
            "DecisionReason": "; ".join(reasons),
            "DecisionRule": "Focus score = 50% existing rank + 20% V1/V2/adjusted agreement + 15% modal-score support + 15% validation. Research decision aid, not a guarantee.",
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    tier_order = {"STRONG_FOCUS": 0, "REVIEW": 1, "LOW_PRIORITY": 2, "PASS": 3}
    out["_tier_order"] = out["DecisionTier"].map(tier_order).fillna(9)
    out = out.sort_values(["_tier_order", "DecisionFocusScore", "ExpectedReturnPerUnit"], ascending=[True, False, False]).drop(columns=["_tier_order"])
    out.to_csv(OUT_PATH, index=False)

    print(f"Decision focus rows: {len(out)}")
    display = [c for c in ["DecisionTier", "DecisionFocusScore", "HomeTeam", "AwayTeam", "MarketType", "Selection", "ModelWinProbability", "ExpectedReturnPerUnit", "Agreement", "ModalSupportNote"] if c in out.columns]
    print(out[display].head(15).to_string(index=False))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
