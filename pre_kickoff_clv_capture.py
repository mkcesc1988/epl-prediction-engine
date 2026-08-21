from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bet_tracker import _norm

HISTORY = Path("data/history")
PROCESSED = Path("data/processed")
REAL_LEDGER = HISTORY / "real_money_bet_ledger.csv"
CAPTURE_LOG = HISTORY / "pre_kickoff_capture_log.csv"
ODDS_LATEST = PROCESSED / "market_odds_latest.csv"

TARGETS = {"T60": 60.0, "T15": 15.0}
WINDOWS = {"T60": (54.0, 66.0), "T15": (8.0, 22.0)}


def _load_log() -> pd.DataFrame:
    if CAPTURE_LOG.exists() and CAPTURE_LOG.stat().st_size:
        return pd.read_csv(CAPTURE_LOG)
    return pd.DataFrame(columns=["CaptureKey", "Label", "CapturedUTC", "MinutesBeforeKickoff", "Fixtures"])


def _open_fixtures(now: pd.Timestamp) -> pd.DataFrame:
    if not REAL_LEDGER.exists() or not REAL_LEDGER.stat().st_size:
        return pd.DataFrame()
    ledger = pd.read_csv(REAL_LEDGER)
    if ledger.empty:
        return ledger
    ledger["KickoffUTC"] = pd.to_datetime(ledger["KickoffUTC"], utc=True, errors="coerce")
    ledger = ledger[(ledger["Result"].astype(str) == "OPEN") & ledger["KickoffUTC"].notna()].copy()
    ledger["MinutesBeforeKickoff"] = (ledger["KickoffUTC"] - now).dt.total_seconds() / 60.0
    return ledger[ledger["MinutesBeforeKickoff"] > 0].copy()


def _cluster_key(rows: pd.DataFrame, label: str) -> str:
    kickoffs = sorted(rows["KickoffUTC"].dt.strftime("%Y-%m-%dT%H:%MZ").unique().tolist())
    return label + "|" + ";".join(kickoffs)


def _due(rows: pd.DataFrame, log: pd.DataFrame, label: str) -> pd.DataFrame:
    lo, hi = WINDOWS[label]
    candidates = rows[(rows["MinutesBeforeKickoff"] >= lo) & (rows["MinutesBeforeKickoff"] <= hi)].copy()
    if candidates.empty:
        return candidates
    keys_done = set(log.get("CaptureKey", pd.Series(dtype=str)).astype(str))
    groups = []
    for kickoff, g in candidates.groupby("KickoffUTC"):
        key = _cluster_key(g, label)
        if key not in keys_done:
            groups.append(g)
    return pd.concat(groups, ignore_index=True) if groups else candidates.iloc[0:0].copy()


def _run(cmd: str) -> None:
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def _benchmark_update(snapshot_utc: pd.Timestamp) -> None:
    if not REAL_LEDGER.exists() or not ODDS_LATEST.exists():
        return
    ledger = pd.read_csv(REAL_LEDGER)
    odds = pd.read_csv(ODDS_LATEST)
    if ledger.empty or odds.empty:
        return

    for col, default in {
        "BenchmarkClosingOdds": np.nan,
        "BenchmarkClosingLine": np.nan,
        "BenchmarkLineMove": np.nan,
        "BenchmarkSource": pd.NA,
        "BenchmarkSnapshotUTC": pd.NA,
        "BenchmarkHoursBeforeKickoff": np.nan,
        "BenchmarkPriceCLV": np.nan,
    }.items():
        if col not in ledger.columns:
            ledger[col] = default

    odds["HomeTeam"] = odds["HomeTeam"].map(_norm)
    odds["AwayTeam"] = odds["AwayTeam"].map(_norm)
    odds["DecimalOdds"] = pd.to_numeric(odds["DecimalOdds"], errors="coerce")
    odds["Point"] = pd.to_numeric(odds.get("Point"), errors="coerce")
    title = odds.get("Bookmaker", pd.Series(index=odds.index, dtype=object)).astype(str).str.lower()
    key = odds.get("BookmakerKey", pd.Series(index=odds.index, dtype=object)).astype(str).str.lower()
    odds = odds[~(title.str.contains("mybookie", na=False) | key.str.contains("mybookie", na=False))].copy()

    now = snapshot_utc
    for idx, bet in ledger.iterrows():
        if str(bet.get("Result", "")) != "OPEN":
            continue
        kickoff = pd.to_datetime(bet.get("KickoffUTC"), utc=True, errors="coerce")
        if pd.isna(kickoff) or now >= kickoff:
            continue
        m = odds[(odds["HomeTeam"] == _norm(bet.get("HomeTeam"))) & (odds["AwayTeam"] == _norm(bet.get("AwayTeam")))].copy()
        mt = str(bet.get("MarketType", "")); sel = str(bet.get("Selection", "")); line = pd.to_numeric(bet.get("Line"), errors="coerce")
        if mt == "Moneyline":
            m = m[m["Market"].astype(str) == "h2h"]
            target = "Draw" if sel.lower() == "draw" else _norm(sel)
            m = m[m["Outcome"].map(lambda x: "Draw" if str(x).lower() == "draw" else _norm(x)) == target]
            close_line = np.nan
        elif mt == "Spread":
            m = m[m["Market"].astype(str) == "spreads"]
            team = sel.rsplit(" ", 1)[0]
            m = m[m["Outcome"].map(_norm) == _norm(team)]
            if m.empty:
                continue
            close_line = float(m["Point"].dropna().median()) if m["Point"].notna().any() else np.nan
            if np.isfinite(close_line):
                same_line = m[np.isclose(m["Point"], close_line, atol=1e-9)]
                if not same_line.empty:
                    m = same_line
        elif mt == "Total":
            m = m[m["Market"].astype(str) == "totals"]
            side = sel.split()[0].title()
            m = m[m["Outcome"].astype(str).str.title() == side]
            if m.empty:
                continue
            close_line = float(m["Point"].dropna().median()) if m["Point"].notna().any() else np.nan
            if np.isfinite(close_line):
                same_line = m[np.isclose(m["Point"], close_line, atol=1e-9)]
                if not same_line.empty:
                    m = same_line
        else:
            continue
        prices = pd.to_numeric(m["DecimalOdds"], errors="coerce").dropna()
        prices = prices[prices > 1.0]
        if prices.empty:
            continue
        close_odds = float(prices.median())
        entry_odds = float(pd.to_numeric(bet.get("EntryOdds"), errors="coerce"))
        hours = (kickoff - now).total_seconds() / 3600.0
        ledger.at[idx, "BenchmarkClosingOdds"] = close_odds
        ledger.at[idx, "BenchmarkClosingLine"] = close_line
        ledger.at[idx, "BenchmarkSource"] = f"consensus_median_{len(prices)}_quotes"
        ledger.at[idx, "BenchmarkSnapshotUTC"] = now.isoformat()
        ledger.at[idx, "BenchmarkHoursBeforeKickoff"] = hours
        if mt == "Moneyline":
            ledger.at[idx, "BenchmarkPriceCLV"] = entry_odds / close_odds - 1.0
        elif pd.notna(line) and np.isfinite(close_line):
            ledger.at[idx, "BenchmarkLineMove"] = close_line - float(line)
            ledger.at[idx, "BenchmarkPriceCLV"] = entry_odds / close_odds - 1.0 if np.isclose(close_line, float(line)) else np.nan
    ledger.to_csv(REAL_LEDGER, index=False)


def _capture(label: str, rows: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    _run("python market_comparison.py")
    _run("python history_logger.py")
    _benchmark_update(now)
    _run("python bet_tracker.py")
    _run("python real_money_tracker.py")
    key = _cluster_key(rows, label)
    rec = pd.DataFrame([{
        "CaptureKey": key,
        "Label": label,
        "CapturedUTC": now.isoformat(),
        "MinutesBeforeKickoff": float(rows["MinutesBeforeKickoff"].median()),
        "Fixtures": "; ".join(sorted((rows["HomeTeam"].astype(str) + " vs " + rows["AwayTeam"].astype(str)).unique())),
    }])
    log = pd.concat([log, rec], ignore_index=True, sort=False).drop_duplicates("CaptureKey", keep="last")
    CAPTURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(CAPTURE_LOG, index=False)
    print(f"Captured {label}: {key}", flush=True)
    return log


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    rows = _open_fixtures(now)
    if rows.empty:
        print("No open tracked fixtures before kickoff.")
        return
    log = _load_log()

    due60 = _due(rows, log, "T60")
    if not due60.empty:
        log = _capture("T60", due60, log)

    now = pd.Timestamp.now(tz="UTC")
    rows = _open_fixtures(now)
    due15 = _due(rows, log, "T15")
    if due15.empty:
        print("No T-15 capture due.")
        return

    log = _capture("T15", due15, log)

    # Keep this one job alive until roughly T-2 so we do not poll the odds API every few minutes.
    earliest = due15["KickoffUTC"].min()
    target = earliest - pd.Timedelta(minutes=2)
    sleep_seconds = max(0.0, (target - pd.Timestamp.now(tz="UTC")).total_seconds())
    if sleep_seconds > 0:
        print(f"Sleeping {sleep_seconds:.0f}s until near-close capture...", flush=True)
        time.sleep(sleep_seconds)

    refreshed = _open_fixtures(pd.Timestamp.now(tz="UTC"))
    near = refreshed[refreshed["KickoffUTC"].isin(due15["KickoffUTC"].unique())].copy()
    if not near.empty:
        log = _capture("T2", near, log)


if __name__ == "__main__":
    main()
