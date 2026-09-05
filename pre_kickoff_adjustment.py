from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
MANUAL_ADJ_PATH = Path("data/manual_availability_adjustments.csv")
OUT_PATH = DATA_DIR / "pre_kickoff_adjusted_latest.csv"


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


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _market_no_vig(odds: pd.DataFrame, date: str, home: str, away: str) -> tuple[float, float, float] | None:
    if odds.empty:
        return None
    required = {"Date", "HomeTeam", "AwayTeam", "Market", "Outcome", "DecimalOdds"}
    if not required.issubset(odds.columns):
        return None

    o = odds.copy()
    o["Date"] = o["Date"].astype(str)
    o["HomeTeam"] = o["HomeTeam"].map(_norm)
    o["AwayTeam"] = o["AwayTeam"].map(_norm)
    q = o[
        (o["Date"] == str(date))
        & (o["HomeTeam"] == _norm(home))
        & (o["AwayTeam"] == _norm(away))
        & (o["Market"] == "h2h")
    ].copy()
    if q.empty:
        return None

    rows = []
    for _, r in q.iterrows():
        price = pd.to_numeric(r.get("DecimalOdds"), errors="coerce")
        if pd.isna(price) or float(price) <= 1.0:
            continue
        outcome = _norm(r.get("Outcome"))
        if str(outcome).lower() in {"draw", "tie"}:
            side = "draw"
        elif outcome == _norm(home):
            side = "home"
        elif outcome == _norm(away):
            side = "away"
        else:
            continue
        rows.append((str(r.get("BookmakerKey", r.get("Bookmaker", "unknown"))), side, float(price)))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["book", "side", "price"])
    probs = []
    for _, g in df.groupby("book"):
        if set(g["side"]) != {"home", "draw", "away"}:
            continue
        raw = {r.side: 1.0 / r.price for r in g.itertuples(index=False)}
        s = sum(raw.values())
        if s <= 0:
            continue
        probs.append([raw["home"] / s, raw["draw"] / s, raw["away"] / s])

    if not probs:
        return None
    arr = np.asarray(probs, dtype=float)
    med = np.median(arr, axis=0)
    med = med / med.sum()
    return float(med[0]), float(med[1]), float(med[2])


def _manual_shift(manual: pd.DataFrame, date: str, home: str, away: str) -> tuple[float, float, float, str]:
    if manual.empty:
        return 0.0, 0.0, 0.0, ""
    m = manual.copy()
    m["Date"] = m["Date"].astype(str)
    m["HomeTeam"] = m["HomeTeam"].map(_norm)
    m["AwayTeam"] = m["AwayTeam"].map(_norm)
    q = m[(m["Date"] == str(date)) & (m["HomeTeam"] == _norm(home)) & (m["AwayTeam"] == _norm(away))]
    if q.empty:
        return 0.0, 0.0, 0.0, ""
    r = q.iloc[-1]
    h = float(np.clip(pd.to_numeric(r.get("HomeProbShift"), errors="coerce") if pd.notna(r.get("HomeProbShift")) else 0.0, -0.05, 0.05))
    d = float(np.clip(pd.to_numeric(r.get("DrawProbShift"), errors="coerce") if pd.notna(r.get("DrawProbShift")) else 0.0, -0.05, 0.05))
    a = float(np.clip(pd.to_numeric(r.get("AwayProbShift"), errors="coerce") if pd.notna(r.get("AwayProbShift")) else 0.0, -0.05, 0.05))
    note = str(r.get("Note", ""))
    return h, d, a, note


def _agreement_label(v1: np.ndarray, v2: np.ndarray, final: np.ndarray) -> str:
    v1_pick = int(np.argmax(v1))
    v2_pick = int(np.argmax(v2))
    final_pick = int(np.argmax(final))
    if v1_pick == v2_pick == final_pick:
        return "STRONG_AGREEMENT"
    if v1_pick == final_pick or v2_pick == final_pick:
        return "PARTIAL_AGREEMENT"
    return "DISAGREEMENT"


def main() -> None:
    pred = _safe_read(DATA_DIR / "daily_predictions_latest.csv")
    v2 = _safe_read(DATA_DIR / "v2_shadow_predictions_latest.csv")
    odds = _safe_read(DATA_DIR / "market_odds_latest.csv")
    manual = _safe_read(MANUAL_ADJ_PATH)

    if pred.empty:
        raise RuntimeError("daily_predictions_latest.csv is empty")

    v2_map = {}
    if not v2.empty:
        for _, r in v2.iterrows():
            key = (str(r["Date"]), _norm(r["HomeTeam"]), _norm(r["AwayTeam"]))
            v2_map[key] = r

    rows = []
    for _, r in pred.iterrows():
        date = str(r["Date"])
        home = _norm(r["HomeTeam"])
        away = _norm(r["AwayTeam"])
        key = (date, home, away)

        v1 = np.array([float(r["P_HomeWin"]), float(r["P_Draw"]), float(r["P_AwayWin"])], dtype=float)
        vr = v2_map.get(key)
        if vr is not None and all(c in vr for c in ["V2_P_HomeWin", "V2_P_Draw", "V2_P_AwayWin"]):
            v2p = np.array([float(vr["V2_P_HomeWin"]), float(vr["V2_P_Draw"]), float(vr["V2_P_AwayWin"])], dtype=float)
        else:
            v2p = v1.copy()

        market = _market_no_vig(odds, date, home, away)
        if market is None:
            mp = v1.copy()
            weights = np.array([0.80, 0.20, 0.0])
            market_status = "NO_MARKET_H2H"
        else:
            mp = np.array(market, dtype=float)
            weights = np.array([0.65, 0.20, 0.15])
            market_status = "MARKET_INCLUDED"

        base = weights[0] * v1 + weights[1] * v2p + weights[2] * mp
        h_shift, d_shift, a_shift, note = _manual_shift(manual, date, home, away)
        shifted = base + np.array([h_shift, d_shift, a_shift], dtype=float)
        shifted = np.clip(shifted, 0.01, None)
        final = shifted / shifted.sum()

        rows.append({
            "Date": date,
            "KickoffUTC": r.get("KickoffUTC"),
            "HomeTeam": home,
            "AwayTeam": away,
            "V1_P_Home": v1[0], "V1_P_Draw": v1[1], "V1_P_Away": v1[2],
            "V2_P_Home": v2p[0], "V2_P_Draw": v2p[1], "V2_P_Away": v2p[2],
            "Market_P_Home": mp[0] if market is not None else np.nan,
            "Market_P_Draw": mp[1] if market is not None else np.nan,
            "Market_P_Away": mp[2] if market is not None else np.nan,
            "HomeAvailabilityShift": h_shift,
            "DrawAvailabilityShift": d_shift,
            "AwayAvailabilityShift": a_shift,
            "AvailabilityNote": note,
            "Adjusted_P_Home": final[0],
            "Adjusted_P_Draw": final[1],
            "Adjusted_P_Away": final[2],
            "AdjustedPick": [home, "Draw", away][int(np.argmax(final))],
            "AdjustedPickProbability": float(np.max(final)),
            "Agreement": _agreement_label(v1, v2p, final),
            "MarketStatus": market_status,
            "Method": "Shadow ensemble: 65% V1.2 + 20% V2 context + 15% no-vig market when available; optional capped manual availability shifts",
            "ProductionEligible": False,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Pre-kickoff adjusted rows: {len(out)}")
    print(out[["Date", "HomeTeam", "AwayTeam", "AdjustedPick", "AdjustedPickProbability", "Agreement", "MarketStatus"]].to_string(index=False))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
