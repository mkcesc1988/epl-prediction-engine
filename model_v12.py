from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from model_v11 import _derived_markets, _score_matrix


@dataclass
class StrengthFit:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    intercept: float
    # Ratings used for teams the fit has never seen (e.g. a promoted side
    # before its first EPL match). Defaults keep the legacy behaviour (0.0).
    unknown_attack: float = 0.0
    unknown_defense: float = 0.0
    promoted_teams: frozenset = field(default_factory=frozenset)

    def team_attack(self, team: str) -> float:
        return float(self.attack.get(team, self.unknown_attack))

    def team_defense(self, team: str) -> float:
        return float(self.defense.get(team, self.unknown_defense))


def effective_home_advantage(fit: StrengthFit, cfg: dict) -> float:
    """Fitted home advantage (log scale) after the configured shrink.

    ``model_v12.home_advantage_shrink`` is a fraction in [0, 1]: 0 keeps the
    fitted value unchanged (default), 1 removes home advantage entirely.
    """
    shrink = float(cfg.get("model_v12", {}).get("home_advantage_shrink", 0.0) or 0.0)
    shrink = min(max(shrink, 0.0), 1.0)
    return float(fit.home_advantage) * (1.0 - shrink)


def _weights(dates: pd.Series, reference_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age_days = (reference_date - pd.to_datetime(dates)).dt.days.clip(lower=0).to_numpy(dtype=float)
    if half_life_days <= 0:
        return np.ones_like(age_days)
    return np.power(0.5, age_days / half_life_days)


def season_start_year(dates) -> np.ndarray:
    """EPL season start year for each date (seasons start in July/August)."""
    d = pd.to_datetime(pd.Series(dates))
    return np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1).astype(int)


def promoted_team_prior(history: pd.DataFrame, reference_date: pd.Timestamp, cfg: dict) -> tuple[float, float, frozenset]:
    """Leakage-safe prior ratings for newly promoted teams.

    A team is "promoted" for the season containing ``reference_date`` when it
    played no EPL match in the previous season. The prior is estimated from
    completed seasons in ``history`` only: the log ratio of promoted teams'
    first-season xG for/against per match versus the league per-team mean.
    Falls back to the configured values when there is not enough history.

    Returns (attack_prior, defense_prior, promoted_teams_this_season).
    Defense follows the model convention: higher = concedes less.
    """
    mc = cfg.get("model_v12", {})
    fb_att = float(mc.get("promoted_prior_attack", -0.20))
    fb_def = float(mc.get("promoted_prior_defense", -0.20))
    min_team_seasons = int(mc.get("promoted_prior_min_team_seasons", 3))
    if history.empty:
        return fb_att, fb_def, frozenset()

    h = history.dropna(subset=["HomeTeam", "AwayTeam", "Date"]).copy()
    h["_season"] = season_start_year(h["Date"])
    ref_season = int(season_start_year([reference_date])[0])

    teams_by_season: dict[int, set] = {}
    for s, g in h.groupby("_season"):
        teams_by_season[int(s)] = set(g["HomeTeam"]).union(g["AwayTeam"])

    prev = teams_by_season.get(ref_season - 1, set())
    current = teams_by_season.get(ref_season, set())
    promoted_now = frozenset(current - prev) if prev else frozenset()

    if not bool(mc.get("promoted_prior_auto", True)):
        return fb_att, fb_def, promoted_now

    if not {"Home_xG", "Away_xG"}.issubset(h.columns):
        return fb_att, fb_def, promoted_now

    att_logs, def_logs = [], []
    xg = h.dropna(subset=["Home_xG", "Away_xG"])
    for s in sorted(teams_by_season):
        if s >= ref_season or (s - 1) not in teams_by_season:
            continue
        new_teams = teams_by_season[s] - teams_by_season[s - 1]
        g = xg[xg["_season"] == s]
        if g.empty or not new_teams:
            continue
        league_mean = float(np.r_[g["Home_xG"], g["Away_xG"]].mean())
        if league_mean <= 0:
            continue
        for t in new_teams:
            home, away = g[g["HomeTeam"] == t], g[g["AwayTeam"] == t]
            n = len(home) + len(away)
            if n < 10:
                continue
            xg_for = (home["Home_xG"].sum() + away["Away_xG"].sum()) / n
            xg_against = (home["Away_xG"].sum() + away["Home_xG"].sum()) / n
            att_logs.append(math.log(max(xg_for, 0.05) / league_mean))
            def_logs.append(-math.log(max(xg_against, 0.05) / league_mean))

    if len(att_logs) < min_team_seasons:
        return fb_att, fb_def, promoted_now
    return float(np.mean(att_logs)), float(np.mean(def_logs)), promoted_now


def _fit_strengths(history: pd.DataFrame, reference_date: pd.Timestamp, cfg: dict) -> StrengthFit | None:
    mc = cfg.get("model_v12", {})
    min_history = int(mc.get("min_history_matches", 200))
    if len(history) < min_history:
        return None

    prior_enabled = bool(mc.get("promoted_prior_enabled", False))
    if prior_enabled:
        p_att, p_def, promoted_now = promoted_team_prior(history, reference_date, cfg)
    else:
        p_att, p_def, promoted_now = 0.0, 0.0, frozenset()

    window = int(mc.get("fit_window_matches", 1140))
    if window > 0 and len(history) > window:
        history = history.tail(window).copy()

    history = history.dropna(subset=["Home_xG", "Away_xG", "HomeTeam", "AwayTeam", "Date"]).copy()
    if len(history) < min_history:
        return None

    teams = sorted(set(history["HomeTeam"]).union(set(history["AwayTeam"])))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    home_idx = history["HomeTeam"].map(idx).to_numpy(dtype=int)
    away_idx = history["AwayTeam"].map(idx).to_numpy(dtype=int)
    y_h = history["Home_xG"].to_numpy(dtype=float)
    y_a = history["Away_xG"].to_numpy(dtype=float)
    w = _weights(history["Date"], reference_date, float(mc.get("half_life_days", 365.0)))

    reg = float(mc.get("regularization", 0.20))

    # Promoted-team prior: shrink promoted teams toward the promoted-team mean
    # instead of the league average, with a strength worth N pseudo-matches.
    # For a log-rate parameter the Poisson information per match is ~lambda,
    # so N matches correspond to a quadratic penalty of 0.5 * N * mean_xg.
    prior_att = np.zeros(n)
    prior_def = np.zeros(n)
    prior_w = np.zeros(n)
    if prior_enabled and promoted_now:
        pseudo = float(mc.get("promoted_prior_matches", 8.0))
        for t in promoted_now:
            if t in idx:
                prior_att[idx[t]] = p_att
                prior_def[idx[t]] = p_def
                prior_w[idx[t]] = 0.5 * pseudo

    # attack[0:n], defense[n:2n], home_advantage, intercept
    x0 = np.zeros(2 * n + 2, dtype=float)
    mean_xg = max(float(np.average(np.r_[y_h, y_a], weights=np.r_[w, w])), 0.15)
    x0[-1] = math.log(mean_xg)
    x0[-2] = math.log(max(float(np.average(y_h, weights=w)), 0.15) / max(float(np.average(y_a, weights=w)), 0.15))

    def objective(theta: np.ndarray) -> float:
        attack = theta[:n]
        defense = theta[n:2*n]
        home_adv = theta[-2]
        intercept = theta[-1]

        # Identifiability: center attack and defense around zero.
        attack_c = attack - attack.mean()
        defense_c = defense - defense.mean()

        eta_h = intercept + home_adv + attack_c[home_idx] - defense_c[away_idx]
        eta_a = intercept + attack_c[away_idx] - defense_c[home_idx]
        lam_h = np.exp(np.clip(eta_h, -4.0, 3.0))
        lam_a = np.exp(np.clip(eta_a, -4.0, 3.0))

        # Poisson quasi-likelihood works for non-negative continuous xG targets
        # up to an additive constant independent of the parameters.
        nll_h = np.sum(w * (lam_h - y_h * np.log(lam_h)))
        nll_a = np.sum(w * (lam_a - y_a * np.log(lam_a)))
        penalty = reg * (np.sum(attack_c ** 2) + np.sum(defense_c ** 2) + home_adv ** 2)
        if prior_w.any():
            scale = math.exp(intercept)
            penalty += scale * np.sum(prior_w * ((attack_c - prior_att) ** 2 + (defense_c - prior_def) ** 2))
        return float(nll_h + nll_a + penalty)

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9})
    if not result.success and not np.isfinite(result.fun):
        return None

    theta = result.x
    attack = theta[:n] - theta[:n].mean()
    defense = theta[n:2*n] - theta[n:2*n].mean()

    return StrengthFit(
        teams=teams,
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_advantage=float(theta[-2]),
        intercept=float(theta[-1]),
        unknown_attack=p_att if prior_enabled else 0.0,
        unknown_defense=p_def if prior_enabled else 0.0,
        promoted_teams=promoted_now,
    )


def _estimate_rho_from_history(history: pd.DataFrame, fit: StrengthFit, cfg: dict) -> float:
    mc = cfg.get("model_v12", {})
    fallback = float(mc.get("rho_fallback", -0.08))
    min_matches = int(mc.get("rho_min_matches", 200))
    if len(history) < min_matches:
        return fallback

    recent = history.tail(int(mc.get("rho_window_matches", 760))).copy()
    home_adv = effective_home_advantage(fit, cfg)
    rows = []
    for _, r in recent.iterrows():
        h, a = str(r["HomeTeam"]), str(r["AwayTeam"])
        ah = fit.team_attack(h)
        dh = fit.team_defense(h)
        aa = fit.team_attack(a)
        da = fit.team_defense(a)
        lam_h = math.exp(np.clip(fit.intercept + home_adv + ah - da, -4.0, 3.0))
        lam_a = math.exp(np.clip(fit.intercept + aa - dh, -4.0, 3.0))
        if pd.notna(r.get("FTHG")) and pd.notna(r.get("FTAG")):
            rows.append((lam_h, lam_a, int(r["FTHG"]), int(r["FTAG"])))

    if len(rows) < min_matches:
        return fallback

    lo = float(mc.get("rho_grid_min", -0.20))
    hi = float(mc.get("rho_grid_max", 0.05))
    step = float(mc.get("rho_grid_step", 0.005))

    def tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1.0 - lh * la * rho
        if x == 0 and y == 1:
            return 1.0 + lh * rho
        if x == 1 and y == 0:
            return 1.0 + la * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    best_rho = fallback
    best_ll = -1e100
    for rho in np.arange(lo, hi + step / 2, step):
        ll = 0.0
        valid = True
        for lh, la, hg, ag in rows:
            t = tau(hg, ag, lh, la, float(rho))
            if t <= 0:
                valid = False
                break
            # Independent Poisson terms do not depend on rho, so only tau is needed.
            ll += math.log(t)
        if valid and ll > best_ll:
            best_ll = ll
            best_rho = float(rho)
    return best_rho


def build_predictions_v12(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Walk-forward xG attack/defense strength model with Dixon-Coles markets.

    Parameters are fitted only on matches strictly earlier than the prediction
    date. All matches played on the same date share the same pre-match fit.
    """
    df = master.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    mc = cfg.get("model_v12", {})
    floor = float(mc.get("lambda_floor", 0.15))
    cap = float(mc.get("lambda_cap", 4.5))
    max_goal = int(mc.get("max_goal", 10))

    outputs: list[dict] = []
    history = df.iloc[0:0].copy()

    for match_date, day_games in df.groupby("Date", sort=True):
        fit = _fit_strengths(history, pd.Timestamp(match_date), cfg)
        rho = _estimate_rho_from_history(history, fit, cfg) if fit is not None else np.nan
        home_adv = effective_home_advantage(fit, cfg) if fit is not None else np.nan

        for _, r in day_games.iterrows():
            row = dict(r)
            h, a = str(r["HomeTeam"]), str(r["AwayTeam"])

            if fit is None:
                lam_h = lam_a = np.nan
            else:
                ah = fit.team_attack(h)
                dh = fit.team_defense(h)
                aa = fit.team_attack(a)
                da = fit.team_defense(a)
                lam_h = math.exp(np.clip(fit.intercept + home_adv + ah - da, -4.0, 3.0))
                lam_a = math.exp(np.clip(fit.intercept + aa - dh, -4.0, 3.0))
                lam_h = max(floor, min(float(lam_h), cap))
                lam_a = max(floor, min(float(lam_a), cap))

            if pd.notna(lam_h) and pd.notna(lam_a):
                matrix = _score_matrix(float(lam_h), float(lam_a), float(rho), max_goal)
                markets = _derived_markets(matrix)
                p_over = float(markets["P_Over_2_5_DC"])
                row.update({
                    "Lambda_Home_xG": lam_h,
                    "Lambda_Away_xG": lam_a,
                    "Lambda_Total_xG": lam_h + lam_a,
                    "DixonColes_Rho": float(rho),
                    "RawP_Over2_5_xG": p_over,
                    "RawP_Under2_5_xG": 1.0 - p_over,
                    "FairOdds_Over2_5_xG": 1.0 / p_over,
                    "FairOdds_Under2_5_xG": 1.0 / (1.0 - p_over),
                    "HomeAttackRating": fit.team_attack(h),
                    "HomeDefenseRating": fit.team_defense(h),
                    "AwayAttackRating": fit.team_attack(a),
                    "AwayDefenseRating": fit.team_defense(a),
                    "HomeAdvantageLog": fit.home_advantage,
                    "HomeAdvantageUsed": home_adv,
                })
                row.update(markets)
            else:
                for key in [
                    "Lambda_Home_xG", "Lambda_Away_xG", "Lambda_Total_xG",
                    "DixonColes_Rho", "RawP_Over2_5_xG", "RawP_Under2_5_xG",
                    "FairOdds_Over2_5_xG", "FairOdds_Under2_5_xG",
                    "HomeAttackRating", "HomeDefenseRating", "AwayAttackRating",
                    "AwayDefenseRating", "HomeAdvantageLog", "HomeAdvantageUsed",
                ]:
                    row[key] = np.nan
            outputs.append(row)

        # Update only after every game on this date has been predicted.
        history = pd.concat([history, day_games], ignore_index=True)

    return pd.DataFrame(outputs)
