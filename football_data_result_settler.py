from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from bet_tracker import _norm, _settle_one

HISTORY = Path("data/history")
PAPER_LEDGER = HISTORY / "auto_bet_ledger.csv"
REAL_LEDGER = HISTORY / "real_money_bet_ledger.csv"
FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"


def _season_code(date_value: object) -> str:
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return ""
    year = dt.year if dt.month >= 7 else dt.year - 1
    return f"{str(year)[-2:]}{str(year + 1)[-2:]}"


def _fetch_results(season_codes: set[str]) -> dict[tuple[str, str], tuple[int, int]]:
    """Best-effort Football-Data fallback.

    This source is supplementary only. A bad/missing/malformed season file must never
    abort settlement because manual results, FPL, and API-Football are independent
    sources used later in the workflow.
    """
    out: dict[tuple[str, str], tuple[int, int]] = {}
    needed = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}

    for code in sorted(c for c in season_codes if c):
        url = FD_URL.format(season=code)
        try:
            r = requests.get(
                url,
                timeout=45,
                headers={"User-Agent": "Mozilla/5.0 EPL prediction engine"},
            )
            r.raise_for_status()

            text = r.text or ""
            content_type = str(r.headers.get("content-type", "")).lower()
            first_line = text.splitlines()[0] if text.splitlines() else ""

            # Football-Data occasionally returns an HTML/error page with HTTP 200.
            if not text.strip():
                print(f"WARNING: Football-Data {code} returned an empty response; skipping")
                continue
            if "<html" in text[:500].lower() or "<!doctype" in text[:500].lower():
                print(f"WARNING: Football-Data {code} returned HTML instead of CSV; skipping")
                continue
            if "," not in first_line and "text/csv" not in content_type and "application/csv" not in content_type:
                print(f"WARNING: Football-Data {code} response does not look like CSV; skipping")
                continue

            try:
                df = pd.read_csv(StringIO(text))
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
                print(f"WARNING: Football-Data {code} CSV could not be parsed: {exc}; skipping")
                continue

            if not needed.issubset(df.columns):
                print(
                    f"WARNING: Football-Data {code} missing expected columns "
                    f"{sorted(needed - set(df.columns))}; skipping"
                )
                continue

            for _, row in df.dropna(subset=["FTHG", "FTAG"]).iterrows():
                try:
                    h, a = _norm(row["HomeTeam"]), _norm(row["AwayTeam"])
                    out[(h, a)] = (int(row["FTHG"]), int(row["FTAG"]))
                except (TypeError, ValueError):
                    continue

        except requests.RequestException as exc:
            print(f"WARNING: Football-Data {code} unavailable: {exc}; skipping")
            continue
        except Exception as exc:
            # This is a tertiary fallback. Never block the primary settlement pipeline.
            print(f"WARNING: Football-Data {code} unexpected error: {exc}; skipping")
            continue

    return out


def _profit(result: str, stake: float, odds: float) -> float:
    if result == "W":
        return stake * (odds - 1.0)
    if result == "L":
        return -stake
    if result == "PUSH":
        return 0.0
    return float("nan")


def _settle_ledger(path: Path, id_col: str, stake_col: str, odds_col: str, profit_col: str, results: dict) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if df.empty or "Result" not in df.columns:
        return 0
    changed = 0
    now = pd.Timestamp.now(tz="UTC").isoformat()
    for idx, bet in df[df["Result"].astype(str) == "OPEN"].iterrows():
        key = (_norm(bet.get("HomeTeam")), _norm(bet.get("AwayTeam")))
        score = results.get(key)
        if score is None:
            continue
        hg, ag = score
        result, _ = _settle_one(bet, hg, ag)
        if result == "OPEN":
            continue
        stake = float(pd.to_numeric(bet.get(stake_col), errors="coerce"))
        odds = float(pd.to_numeric(bet.get(odds_col), errors="coerce"))
        df.at[idx, "Result"] = result
        df.at[idx, "FinalScore"] = f"{hg}-{ag}"
        df.at[idx, profit_col] = _profit(result, stake, odds)
        df.at[idx, "SettledUTC"] = now
        changed += 1
        print(f"Football-Data settled {bet.get(id_col)}: {key[0]} vs {key[1]} {hg}-{ag} -> {result}")
    if changed:
        df.to_csv(path, index=False)
    return changed


def main() -> None:
    frames = []
    for path in [PAPER_LEDGER, REAL_LEDGER]:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                print(f"WARNING: could not read {path}: {exc}")

    codes: set[str] = set()
    for df in frames:
        if "Date" in df.columns:
            codes.update(_season_code(v) for v in df["Date"].dropna())

    results = _fetch_results(codes)
    paper = _settle_ledger(PAPER_LEDGER, "BetID", "StakeUnits", "EntryOdds", "ProfitUnits", results)
    real = _settle_ledger(REAL_LEDGER, "RealBetID", "StakeUSD", "EntryOdds", "ProfitUSD", results)
    print(f"Football-Data settlements: paper={paper}, real_money={real}")


if __name__ == "__main__":
    main()
