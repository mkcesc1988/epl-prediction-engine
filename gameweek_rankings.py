from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model_v11 import _score_matrix
from pipeline import load_config, normalize_team

DATA_DIR = Path("data/processed")


def _is_mybookie(df: pd.DataFrame) -> pd.Series:
    title = df.get("Bookmaker", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    key = df.get("BookmakerKey", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    return title.str.contains("mybookie", na=False) | key.str.contains("mybookie", na=False)


def _quote_timestamp(df: pd.DataFrame) -> pd.Series:
    last_update = pd.to_datetime(
        df.get("BookLastUpdate", pd.Series(index=df.index, dtype=object)),
        utc=True,
        errors="coerce",
    )
    captured = pd.to_datetime(
        df.get("QuoteCapturedAt", pd.Series(index=df.index, dtype=object)),
        utc=True,
        errors="coerce",
    )
    return last_update.fillna(captured)


def _fresh_mybookie_quotes(df: pd.DataFrame, cfg: dict, now_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    max_age = float(
        cfg.get("market_comparison", {}).get("executable_quote_max_age_minutes", 240)
    )
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    quote_time = _quote_timestamp(out)
    age_minutes = (now - quote_time).dt.total_seconds() / 60.0
    fresh_mask = quote_time.notna() & age_minutes.ge(-5.0) & age_minutes.le(max_age)

    out["PriceTimestampUTC"] = quote_time
    out["PriceAgeMinutes"] = age_minutes
    out["PriceFreshnessStatus"] = np.where(fresh_mask, "FRESH", "STALE")
    out = out.loc[fresh_mask].copy()
    if out.empty:
        return out

    source = out.get("PriceSource", pd.Series(index=out.index, dtype=object)).astype(str)
    out["_SourcePriority"] = source.map({
        "odds_api_mybookie_direct": 2,
        "odds_api_region": 1,
        "manual_screenshot": 0,
    }).fillna(0)

    dedupe = ["Date", "HomeTeam", "AwayTeam", "Market", "Outcome", "Point"]
    out = (
        out.sort_values(
            ["PriceTimestampUTC", "_SourcePriority", "DecimalOdds"],
            ascending=[False, False, False],
            na_position="last",
        )
        .drop_duplicates(dedupe, keep="first")
        .drop(columns=["_SourcePriority"])
        .reset_index(drop=True)
    )
    return out


def _norm(name: object) -> str:
    mapping = {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton and Hove Albion": "Brighton",
        "Nottingham Forest": "Nott'm Forest",
        "Leeds United": "Leeds",
    }
    value = str(name).strip()
    return normalize_team(mapping.get(value, value))


def _book_id(df: pd.DataFrame) -> pd.Series:
    key = df.get("BookmakerKey", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()
    title = df.get("Bookmaker", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()
    return key.where(key.ne(""), title)


def _consensus_no_vig_probability(q: pd.Series, odds: pd.DataFrame) -> tuple[float, int, str]:
    """Median no-vig probability for the quoted selection across available books.

    For 1X2 we normalize the three implied probabilities within each book.
    For totals and spreads we normalize the two opposing sides at the same line.
    Non-MyBookie books are preferred so the executable price is not also the
    primary prior. If none are available, a de-vigged MyBookie pair is used.
    """
    if odds.empty:
        return np.nan, 0, "unavailable"

    work = odds.copy()
    work["Date"] = work["Date"].astype(str)
    work["HomeTeam"] = work["HomeTeam"].map(_norm)
    work["AwayTeam"] = work["AwayTeam"].map(_norm)
    work["_BookId"] = _book_id(work)
    work["_Price"] = pd.to_numeric(work.get("DecimalOdds"), errors="coerce")
    work["_Point"] = pd.to_numeric(work.get("Point"), errors="coerce")
    work = work[
        work["Date"].eq(str(q.get("Date", "")))
        & work["HomeTeam"].eq(_norm(q.get("HomeTeam")))
        & work["AwayTeam"].eq(_norm(q.get("AwayTeam")))
        & work["Market"].astype(str).eq(str(q.get("Market", "")))
        & work["_Price"].gt(1.0)
    ].copy()
    if work.empty:
        return np.nan, 0, "unavailable"

    target_outcome = str(q.get("Outcome", ""))
    target_norm = _norm(target_outcome)
    point_raw = pd.to_numeric(q.get("Point"), errors="coerce")
    point = float(point_raw) if pd.notna(point_raw) else np.nan
    market = str(q.get("Market", ""))
    home = _norm(q.get("HomeTeam"))
    away = _norm(q.get("AwayTeam"))

    def collect(pool: pd.DataFrame) -> list[float]:
        probs: list[float] = []
        for _, g in pool.groupby("_BookId", dropna=False):
            if market == "h2h":
                prices: dict[str, float] = {}
                for _, r in g.iterrows():
                    label = str(r.get("Outcome", ""))
                    key = "draw" if label.lower() == "draw" else _norm(label)
                    prices[key] = float(r["_Price"])
                needed = [home, away, "draw"]
                if not all(k in prices for k in needed):
                    continue
                implied = {k: 1.0 / prices[k] for k in needed}
                denom = sum(implied.values())
                target_key = "draw" if target_outcome.lower() == "draw" else target_norm
                if target_key in implied and denom > 0:
                    probs.append(implied[target_key] / denom)

            elif market == "totals" and pd.notna(point):
                same = g[np.isclose(g["_Point"], point, atol=1e-9, equal_nan=False)].copy()
                prices = {str(r.get("Outcome", "")).title(): float(r["_Price"]) for _, r in same.iterrows()}
                if "Over" not in prices or "Under" not in prices:
                    continue
                implied_over, implied_under = 1.0 / prices["Over"], 1.0 / prices["Under"]
                denom = implied_over + implied_under
                side = target_outcome.title()
                if denom > 0 and side in {"Over", "Under"}:
                    probs.append((implied_over if side == "Over" else implied_under) / denom)

            elif market == "spreads" and pd.notna(point):
                opponent = away if target_norm == home else home if target_norm == away else ""
                if not opponent:
                    continue
                selected = g[
                    g["Outcome"].map(_norm).eq(target_norm)
                    & np.isclose(g["_Point"], point, atol=1e-9, equal_nan=False)
                ]
                opposite = g[
                    g["Outcome"].map(_norm).eq(opponent)
                    & np.isclose(g["_Point"], -point, atol=1e-9, equal_nan=False)
                ]
                if selected.empty or opposite.empty:
                    continue
                p1 = 1.0 / float(selected.iloc[-1]["_Price"])
                p2 = 1.0 / float(opposite.iloc[-1]["_Price"])
                if p1 + p2 > 0:
                    probs.append(p1 / (p1 + p2))
        return probs

    non_my = work.loc[~_is_mybookie(work)].copy()
    probs = collect(non_my)
    source = "consensus_non_mybookie"
    if not probs:
        probs = collect(work.loc[_is_mybookie(work)].copy())
        source = "mybookie_devig_fallback"
    if not probs:
        return np.nan, 0, "unavailable"
    return float(np.median(probs)), int(len(probs)), source


def _decision_probability(
    p_win: float,
    p_push: float,
    market_prior: float,
    market_label: str,
    cfg: dict,
) -> tuple[float, float]:
    """Shrink model probability toward the no-vig market prior.

    For push-capable lines, the blend is performed on the win probability
    conditional on the bet not pushing, then converted back to unconditional
    win probability for EV/Kelly calculations.
    """
    dc = cfg.get("decision", {})
    if not bool(dc.get("market_anchor_enabled", True)) or not np.isfinite(market_prior):
        effective = p_win / max(1e-12, 1.0 - p_push)
        return float(p_win), float(effective)

    weights = {
        "Moneyline": float(dc.get("market_anchor_weight_moneyline", 0.45)),
        "Spread": float(dc.get("market_anchor_weight_spread", 0.40)),
        "Total": float(dc.get("market_anchor_weight_total", 0.35)),
    }
    weight = float(np.clip(weights.get(market_label, 0.40), 0.0, 1.0))
    nonpush_mass = max(1e-12, 1.0 - p_push)
    raw_effective = float(np.clip(p_win / nonpush_mass, 0.0, 1.0))
    prior = float(np.clip(market_prior, 0.0, 1.0))
    decision_effective = (1.0 - weight) * raw_effective + weight * prior
    decision_win = decision_effective * nonpush_mass
    return float(decision_win), float(decision_effective)


def _decision_status(edge: float, consensus_books: int, raw_gap: float, cfg: dict) -> str:
    dc = cfg.get("decision", {})
    no_bet = float(dc.get("no_bet_edge", 0.03))
    eligible = float(dc.get("bet_eligible_edge", 0.05))
    min_books = int(dc.get("min_consensus_books", 2))
    max_raw_gap = float(dc.get("max_raw_model_market_gap", 0.18))

    if not np.isfinite(edge) or edge < no_bet:
        return "NO_BET"
    if edge < eligible:
        return "REVIEW"
    if consensus_books < min_books:
        return "REVIEW_LOW_MARKET_SUPPORT"
    if np.isfinite(raw_gap) and raw_gap > max_raw_gap:
        return "REVIEW_MODEL_MARKET_DISAGREEMENT"
    return "BET_ELIGIBLE"


def _binary_fair_odds(p_win: float, p_push: float = 0.0) -> float:
    if p_win <= 0:
        return np.inf
    return (1.0 - p_push) / p_win


def _ev_with_push(p_win: float, p_push: float, decimal_odds: float) -> float:
    p_loss = max(0.0, 1.0 - p_win - p_push)
    return p_win * (decimal_odds - 1.0) - p_loss


def _totals_prob(matrix: np.ndarray, side: str, line: float) -> tuple[float, float]:
    p_win = 0.0
    p_push = 0.0
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            p = float(matrix[h, a])
            total = h + a
            if side == "Over":
                if total > line:
                    p_win += p
                elif abs(total - line) < 1e-9:
                    p_push += p
            else:
                if total < line:
                    p_win += p
                elif abs(total - line) < 1e-9:
                    p_push += p
    return p_win, p_push


def _spread_prob(matrix: np.ndarray, home_team: str, away_team: str, selection: str, point: float) -> tuple[float, float]:
    selection = _norm(selection)
    is_home = selection == _norm(home_team)
    is_away = selection == _norm(away_team)
    if not (is_home or is_away):
        return np.nan, np.nan

    p_win = 0.0
    p_push = 0.0
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            p = float(matrix[h, a])
            margin = (h - a) if is_home else (a - h)
            adjusted = margin + point
            if adjusted > 1e-9:
                p_win += p
            elif abs(adjusted) <= 1e-9:
                p_push += p
    return p_win, p_push


def _moneyline_prob(pred: pd.Series, selection: str) -> tuple[float, float]:
    s = str(selection).strip().lower()
    home = str(pred["HomeTeam"]).strip().lower()
    away = str(pred["AwayTeam"]).strip().lower()
    if s == home:
        return float(pred["P_HomeWin"]), 0.0
    if s == away:
        return float(pred["P_AwayWin"]), 0.0
    if s in {"draw", "tie"}:
        return float(pred["P_Draw"]), 0.0
    return np.nan, np.nan


def _validation_factor(market_label: str, selection: str) -> tuple[float, str]:
    """Reliability factor based on how directly the market has been calibrated/backtested.

    V1.2's explicit calibration and historical betting evaluation are strongest for
    O/U 2.5. Other markets are derived consistently from the Dixon-Coles score
    matrix but have not yet accumulated equally deep market-specific validation.
    """
    if market_label == "Total" and (selection.startswith("Over 2.5") or selection.startswith("Under 2.5")):
        return 1.00, "Calibrated O/U 2.5"
    if market_label == "Moneyline":
        return 0.88, "Derived, live validation building"
    if market_label == "Spread":
        return 0.84, "Derived, live validation building"
    return 0.86, "Derived alternate total"


def _bet_quality_score(model_p: float, ev: float, validation_factor: float, cfg: dict) -> float:
    rc = cfg.get("ranking", {})
    wp = float(rc.get("quality_probability_weight", 0.50))
    we = float(rc.get("quality_edge_weight", 0.35))
    wv = float(rc.get("quality_validation_weight", 0.15))

    prob_component = np.clip(model_p, 0.0, 1.0)
    # Map roughly -5% to +15% EV onto 0..1, with no assumption that +15% is certain.
    edge_component = np.clip((ev + 0.05) / 0.20, 0.0, 1.0)
    validation_component = np.clip(validation_factor, 0.0, 1.0)
    denom = wp + we + wv if (wp + we + wv) > 0 else 1.0
    return float(100.0 * (
        wp * prob_component + we * edge_component + wv * validation_component
    ) / denom)


def _profitability_score(ev: float, cfg: dict) -> float:
    full = float(cfg.get("ranking", {}).get("profitability_full_score_ev", 0.15))
    if full <= 0:
        full = 0.15
    return float(100.0 * np.clip(ev / full, 0.0, 1.0))


def _overall_score(quality: float, profitability: float, cfg: dict) -> float:
    rc = cfg.get("ranking", {})
    wq = float(rc.get("overall_quality_weight", 0.60))
    wp = float(rc.get("overall_profitability_weight", 0.40))
    denom = wq + wp if (wq + wp) > 0 else 1.0
    return float((wq * quality + wp * profitability) / denom)


def _grade(score: float, ev: float, validation_factor: float) -> str:
    if ev <= 0:
        return "PASS"
    # Do not award A-grade to a market that is still lightly validated.
    if score >= 80 and validation_factor >= 0.95:
        return "A"
    if score >= 72:
        return "B"
    if score >= 62:
        return "C"
    return "D"


def build_rankings(predictions: pd.DataFrame, odds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if predictions.empty or odds.empty:
        return pd.DataFrame()

    all_my = odds[_is_mybookie(odds)].copy()
    if all_my.empty:
        return pd.DataFrame()

    my = _fresh_mybookie_quotes(all_my, cfg)
    if my.empty:
        return pd.DataFrame()

    my["HomeTeam"] = my["HomeTeam"].map(_norm)
    my["AwayTeam"] = my["AwayTeam"].map(_norm)
    my["Date"] = my["Date"].astype(str)

    preds = predictions.copy()
    preds["HomeTeam"] = preds["HomeTeam"].map(_norm)
    preds["AwayTeam"] = preds["AwayTeam"].map(_norm)
    preds["Date"] = preds["Date"].astype(str)

    pred_map = {
        (r["Date"], r["HomeTeam"], r["AwayTeam"]): r
        for _, r in preds.iterrows()
    }

    rows: list[dict] = []
    max_goal = int(cfg.get("model_v12", {}).get("max_goal", 10))
    now_utc = pd.Timestamp.now(tz="UTC")

    for _, q in my.iterrows():
        key = (q["Date"], q["HomeTeam"], q["AwayTeam"])
        pred = pred_map.get(key)
        if pred is None:
            continue

        kickoff = pd.to_datetime(pred.get("KickoffUTC"), utc=True, errors="coerce")
        if pd.notna(kickoff) and kickoff <= now_utc:
            continue

        market = str(q.get("Market", ""))
        outcome = str(q.get("Outcome", ""))
        odds_price = pd.to_numeric(q.get("DecimalOdds"), errors="coerce")
        if pd.isna(odds_price) or float(odds_price) <= 1.0:
            continue
        odds_price = float(odds_price)

        lam_h = float(pred["Lambda_Home_xG"])
        lam_a = float(pred["Lambda_Away_xG"])
        rho = float(pred.get("DixonColes_Rho", -0.08))
        matrix = _score_matrix(lam_h, lam_a, rho, max_goal)

        point_raw = pd.to_numeric(q.get("Point"), errors="coerce")
        point = float(point_raw) if pd.notna(point_raw) else np.nan
        p_win = np.nan
        p_push = 0.0
        market_label = market
        selection = outcome

        if market == "h2h":
            p_win, p_push = _moneyline_prob(pred, outcome)
            market_label = "Moneyline"
        elif market == "totals":
            if pd.isna(point):
                continue
            side = outcome.title()
            if side not in {"Over", "Under"}:
                continue
            if abs(point * 2 - round(point * 2)) > 1e-9:
                continue
            p_win, p_push = _totals_prob(matrix, side, point)
            selection = f"{side} {point:g}"
            market_label = "Total"
            # Use the explicitly calibrated V1.2 O/U 2.5 probability where available.
            if abs(point - 2.5) < 1e-9:
                if side == "Over" and pd.notna(pred.get("CalP_Over2_5")):
                    p_win, p_push = float(pred["CalP_Over2_5"]), 0.0
                elif side == "Under" and pd.notna(pred.get("CalP_Under2_5")):
                    p_win, p_push = float(pred["CalP_Under2_5"]), 0.0
        elif market == "spreads":
            if pd.isna(point):
                continue
            if abs(point * 2 - round(point * 2)) > 1e-9:
                continue
            p_win, p_push = _spread_prob(matrix, pred["HomeTeam"], pred["AwayTeam"], outcome, point)
            selection = f"{_norm(outcome)} {point:+g}"
            market_label = "Spread"
        else:
            continue

        if pd.isna(p_win):
            continue

        p_win = float(p_win)
        p_push = float(p_push)
        raw_fair_odds = _binary_fair_odds(p_win, p_push)
        raw_ev = _ev_with_push(p_win, p_push, odds_price)
        raw_effective = p_win / max(1e-12, 1.0 - p_push)
        implied = 1.0 / odds_price

        market_prior, consensus_books, market_prior_source = _consensus_no_vig_probability(q, odds)
        if not np.isfinite(market_prior):
            # Conservative fallback when a proper paired market is unavailable.
            market_prior = implied
            market_prior_source = "raw_mybookie_implied_fallback"
            consensus_books = 0

        decision_p_win, decision_effective = _decision_probability(
            p_win, p_push, market_prior, market_label, cfg
        )
        fair_odds = _binary_fair_odds(decision_p_win, p_push)
        ev = _ev_with_push(decision_p_win, p_push, odds_price)
        decision_edge = decision_effective - float(market_prior)
        raw_market_gap = abs(raw_effective - float(market_prior))
        decision_status = _decision_status(decision_edge, consensus_books, raw_market_gap, cfg)

        validation_factor, validation_label = _validation_factor(market_label, selection)
        quality = _bet_quality_score(decision_p_win, ev, validation_factor, cfg)
        profitability = _profitability_score(ev, cfg)
        overall = _overall_score(quality, profitability, cfg)

        rows.append({
            "Date": pred["Date"],
            "KickoffUTC": pred.get("KickoffUTC"),
            "FPLGameweek": pred.get("FPLGameweek"),
            "HomeTeam": pred["HomeTeam"],
            "AwayTeam": pred["AwayTeam"],
            "MarketType": market_label,
            "Selection": selection,
            "Line": point if pd.notna(point) else np.nan,
            "Bookmaker": q.get("Bookmaker"),
            "BookmakerKey": q.get("BookmakerKey"),
            "BookLastUpdate": q.get("BookLastUpdate"),
            "QuoteCapturedAt": q.get("QuoteCapturedAt"),
            "PriceSource": q.get("PriceSource"),
            "PriceTimestampUTC": q.get("PriceTimestampUTC"),
            "PriceAgeMinutes": q.get("PriceAgeMinutes"),
            "PriceFreshnessStatus": q.get("PriceFreshnessStatus"),
            "MyBookieOdds": odds_price,
            "RawModelWinProbability": p_win,
            "RawModelEffectiveProbability": raw_effective,
            "RawModelFairOdds": raw_fair_odds,
            "RawExpectedReturnPerUnit": raw_ev,
            "ModelWinProbability": decision_p_win,
            "DecisionEffectiveProbability": decision_effective,
            "PushProbability": p_push,
            "ModelFairOdds": fair_odds,
            "MyBookieImpliedProbability": implied,
            "ConsensusFairProbability": market_prior,
            "ConsensusBookCount": consensus_books,
            "MarketPriorSource": market_prior_source,
            "RawModelMarketGap": raw_market_gap,
            "ProbabilityEdge": decision_edge,
            "DecisionStatus": decision_status,
            "ExpectedReturnPerUnit": ev,
            "ExpectedProfitPer100": ev * 100.0,
            "ValidationFactor": validation_factor,
            "ValidationStatus": validation_label,
            "BetQualityScore": quality,
            "ProfitabilityScore": profitability,
            "OverallRankScore": overall,
            "Grade": _grade(overall, ev, validation_factor),
            "Lambda_Home_xG": lam_h,
            "Lambda_Away_xG": lam_a,
            "MostLikelyScore": pred.get("MostLikelyScore"),
            "ModelVersion": pred.get("ModelVersion", "V1.2"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    dedupe = ["Date", "HomeTeam", "AwayTeam", "MarketType", "Selection"]
    out = out.sort_values("MyBookieOdds", ascending=False).drop_duplicates(dedupe, keep="first")
    out = out.sort_values(["OverallRankScore", "ExpectedReturnPerUnit"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "GameweekRank", np.arange(1, len(out) + 1))
    return out


def main() -> None:
    cfg = load_config()
    pred_path = DATA_DIR / "daily_predictions_latest.csv"
    odds_path = DATA_DIR / "market_odds_latest.csv"

    if not pred_path.exists() or not odds_path.exists():
        raise RuntimeError("Run daily predictions and market comparison before gameweek rankings")

    predictions = pd.read_csv(pred_path)
    odds = pd.read_csv(odds_path)
    ranked = build_rankings(predictions, odds, cfg)

    out_path = DATA_DIR / "gameweek_rankings_latest.csv"
    ranked.to_csv(out_path, index=False)

    print(f"Ranked MyBookie selections: {len(ranked)}")
    if not ranked.empty:
        display = ranked[[
            "GameweekRank", "HomeTeam", "AwayTeam", "MarketType", "Selection",
            "MyBookieOdds", "RawModelWinProbability", "ModelWinProbability", "ConsensusFairProbability",
            "ProbabilityEdge", "DecisionStatus", "ModelFairOdds", "ExpectedReturnPerUnit",
            "ExpectedProfitPer100", "BetQualityScore", "ProfitabilityScore",
            "OverallRankScore", "Grade", "ValidationStatus"
        ]]
        pd.set_option("display.max_columns", None)
        print(display.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
