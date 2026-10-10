"""V1.3 conservative, auditable market decision layer.

Research only. Never treats unverified lineups or old prices as executable bets.
Consumes existing calibrated/market-anchored probabilities. No retrospective
probability fitting is performed on the target gameweek.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
LINEUP_PATH = Path("data/lineup_confirmations_v13.csv")
OUT_PATH = DATA_DIR / "actionable_decisions_v13_latest.csv"

DEFAULTS = {
    "min_ev": 0.05,
    "min_edge": 0.05,
    "min_consensus_books": 2,
    "min_overall_score": 60.0,
    "max_quote_age_minutes": 120.0,
}

def _number(value: object, fallback: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _norm(value: object) -> str:
    mapping = {
        "Nottingham Forest": "Nott'm Forest",
        "Tottenham Hotspur": "Spurs",
        "Tottenham": "Spurs",
        "Manchester United": "Man United",
        "Manchester City": "Man City",
        "Ipswich": "Ipswich Town",
    }
    s = str(value).strip()
    return mapping.get(s, s)


def _lineup_lookup(lineups: pd.DataFrame) -> dict[tuple[str, str, str], bool]:
    if lineups.empty:
        return {}
    required = {"Date", "HomeTeam", "AwayTeam", "LineupsVerified"}
    if not required.issubset(lineups.columns):
        return {}
    lookup = {}
    for _, r in lineups.iterrows():
        yes = str(r["LineupsVerified"]).strip().lower() in {"true", "yes", "1"}
        lookup[(str(r["Date"])[:10], _norm(r["HomeTeam"]), _norm(r["AwayTeam"]))] = yes
    return lookup


def _risk_rule(row: pd.Series) -> str:
    """Pre-registered research exclusions from historical bucket audits.

    Exclusions are hypothesis-generating; they are NOT proof of forward ROI.
    """
    market = str(row.get("MarketType", ""))
    price = _number(row.get("MyBookieOdds"))
    line = _number(row.get("Line"))
    if market == "Moneyline":
        if 2.0 <= price < 2.5:
            return "SUSPEND_ML_PLUS100_149"
        if price >= 4.0:
            return "SUSPEND_ML_PLUS300_OR_LONGER"
        return ""
    if market == "Spread":
        if not np.isfinite(line):
            return "UNSUPPORTED_SPREAD_LINE"
        if line < 0:
            return "SUSPEND_FAVORITE_HANDICAP"
        if line == 0:
            return "RESEARCH_PICKEM_ONLY"
        return ""
    if market == "Total":
        if not np.isfinite(line) or abs(line - 2.5) > 1e-9:
            return "UNVALIDATED_TOTAL_LINE"
        return ""
    return "UNVALIDATED_MARKET"


def _shared_baseline(adjusted: pd.Series | None) -> bool:
    if adjusted is None:
        return True
    a = np.array([_number(adjusted.get(f"V1_P_{k}")) for k in ("Home", "Draw", "Away")])
    b = np.array([_number(adjusted.get(f"V2_P_{k}")) for k in ("Home", "Draw", "Away")])
    return bool(np.isfinite(a).all() and np.isfinite(b).all() and np.max(np.abs(a - b)) < 0.005)


def build_decisions(
    rankings: pd.DataFrame,
    adjusted: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    now_utc: pd.Timestamp | None = None,
    policy: dict | None = None,
) -> pd.DataFrame:
    """Return one preferred *paper* selection per fixture, plus rejected alternatives.

    Candidate requires market odds, decision-edge/EV gates, quoted consensus,
    appropriate historical market category, and minimum historical grade.
    Unverified lineups or stale prices block executable status. We deliberately
    favor winning probability over EV only AFTER positive EV passes all gates.
    """
    if rankings.empty:
        return pd.DataFrame()
    rules = {**DEFAULTS, **(policy or {})}
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    verified = _lineup_lookup(lineups if lineups is not None else pd.DataFrame())
    amap = {}
    if adjusted is not None and not adjusted.empty:
        for _, r in adjusted.iterrows():
            key = (str(r.get("Date"))[:10], _norm(r.get("HomeTeam")), _norm(r.get("AwayTeam")))
            amap[key] = r

    result = []
    for _, r in rankings.iterrows():
        out = r.to_dict()
        key = (str(r.get("Date"))[:10], _norm(r.get("HomeTeam")), _norm(r.get("AwayTeam")))
        price = _number(r.get("MyBookieOdds"))
        p = _number(r.get("DecisionEffectiveProbability", r.get("ModelWinProbability")))
        ev = _number(r.get("ExpectedReturnPerUnit"))
        edge = _number(r.get("ProbabilityEdge"))
        quality = _number(r.get("OverallRankScore"))
        books = _number(r.get("ConsensusBookCount"), 0)
        kickoff = pd.to_datetime(r.get("KickoffUTC"), utc=True, errors="coerce")
        quote = pd.to_datetime(r.get("PriceTimestampUTC"), utc=True, errors="coerce")
        age = (now - quote).total_seconds() / 60.0 if pd.notna(quote) else float("inf")
        confirmed = bool(verified.get(key, False))
        reason = ""
        if pd.notna(kickoff) and kickoff <= now:
            reason = "KICKOFF_PASSED"
        elif not all(np.isfinite(v) for v in (price, p, ev, edge, quality)) or price <= 1:
            reason = "MISSING_PRICE_OR_PROBABILITY"
        elif str(r.get("DecisionStatus")) != "BET_ELIGIBLE":
            reason = "EXISTING_EDGE_GATE_NOT_PASSED"
        elif (edge < rules["min_edge"] or ev < rules["min_ev"] or
              books < rules["min_consensus_books"] or quality < rules["min_overall_score"]):
            reason = "INSUFFICIENT_VALIDATED_VALUE"
        else:
            reason = _risk_rule(r)

        if reason:
            status = "NO_BET"
        elif not confirmed:
            status = "WATCH_LINEUPS_UNVERIFIED"
        elif not np.isfinite(age) or age < -5 or age > rules["max_quote_age_minutes"]:
            status = "REPRICE_REQUIRED"
        else:
            status = "PAPER_CANDIDATE"

        out.update({
            "V13Status": status,
            "V13Reason": reason or ("Starting XIs not explicitly verified" if not confirmed else "Price update required" if status == "REPRICE_REQUIRED" else "Passed paper-only screening"),
            "V13QuoteAgeMinutes": age if np.isfinite(age) else np.nan,
            "V13LineupsVerified": confirmed,
            "V13IndependentV2": not _shared_baseline(amap.get(key)),
            "V13Priority": p,  # after positive-EV filtering, prefer higher win frequency
            "V13PreferredForMatch": False,
            "V13PolicyVersion": "V1.3-gates-1",
        })
        result.append(out)

    frame = pd.DataFrame(result)
    if frame.empty:
        return frame
    active = frame[frame["V13Status"].isin(
        ["WATCH_LINEUPS_UNVERIFIED", "REPRICE_REQUIRED", "PAPER_CANDIDATE"]
    )].copy()
    active = active.sort_values(["V13Priority", "ExpectedReturnPerUnit"], ascending=[False, False])
    chosen = active.drop_duplicates(["Date", "HomeTeam", "AwayTeam"]).index
    frame.loc[chosen, "V13PreferredForMatch"] = True
    other = active.index.difference(chosen)
    frame.loc[other, "V13Status"] = "ALTERNATIVE_CORRELATED"
    frame.loc[other, "V13Reason"] = "One primary selection per fixture; do not stack correlated bets"
    order = {
        "PAPER_CANDIDATE": 0, "WATCH_LINEUPS_UNVERIFIED": 1,
        "REPRICE_REQUIRED": 2, "ALTERNATIVE_CORRELATED": 3, "NO_BET": 4,
    }
    frame["_sort"] = frame["V13Status"].map(order).fillna(8)
    frame = frame.sort_values(["_sort", "V13PreferredForMatch", "V13Priority"],
                              ascending=[True, False, False]).drop(columns="_sort")
    return frame.reset_index(drop=True)


def main() -> None:
    rankings_file = DATA_DIR / "gameweek_rankings_latest.csv"
    if not rankings_file.exists():
        raise FileNotFoundError("Run gameweek_rankings.py before the V1.3 decision gate")
    rankings = pd.read_csv(rankings_file)
    adjusted_path = DATA_DIR / "pre_kickoff_adjusted_latest.csv"
    adj = pd.read_csv(adjusted_path) if adjusted_path.exists() else pd.DataFrame()
    lineups = pd.read_csv(LINEUP_PATH) if LINEUP_PATH.exists() else pd.DataFrame()
    out = build_decisions(rankings, adj, lineups)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    columns = ["Date", "HomeTeam", "AwayTeam", "MarketType", "Selection",
               "MyBookieOdds", "DecisionEffectiveProbability",
               "ExpectedReturnPerUnit", "V13Status", "V13PreferredForMatch"]
    print(out[[c for c in columns if c in out.columns]].head(15).to_string(index=False))
    print(f"Saved {OUT_PATH}; V1.3 is paper-only, not an automated wager")


if __name__ == "__main__":
    main()
