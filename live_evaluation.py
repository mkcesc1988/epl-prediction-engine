from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PREDICTION_KEY = ["Season", "FPLFixtureId", "ModelVersion"]


def _utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def archive_predictions(predictions: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Append first-seen pre-match predictions to an immutable-style CSV ledger.

    A fixture/model version is written only once. Later reruns do not replace the
    original forecast, which prevents hindsight contamination of live evaluation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    incoming = predictions.copy()
    if incoming.empty:
        if path.exists():
            return pd.read_csv(path)
        return incoming

    if "GeneratedAtUTC" not in incoming.columns:
        incoming["GeneratedAtUTC"] = _utc_now_iso()

    missing = [c for c in PREDICTION_KEY if c not in incoming.columns]
    if missing:
        raise ValueError(f"Prediction archive missing key columns: {missing}")

    incoming = incoming.dropna(subset=["FPLFixtureId"]).copy()
    incoming["FPLFixtureId"] = pd.to_numeric(incoming["FPLFixtureId"], errors="coerce").astype("Int64")
    incoming = incoming.dropna(subset=["FPLFixtureId"])

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming

    combined = combined.drop_duplicates(PREDICTION_KEY, keep="first")
    combined = combined.sort_values(["Season", "FPLFixtureId", "ModelVersion"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def settle_predictions(archive: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attach official completed-match outcomes to archived forecasts."""
    if archive.empty:
        return archive.copy()

    out = archive.copy()
    if "FPLFixtureId" not in out.columns:
        raise ValueError("Archive requires FPLFixtureId for leakage-safe settlement")

    fx = fixtures.copy()
    required = {"FPLFixtureId", "Finished", "HomeGoals", "AwayGoals"}
    missing = required.difference(fx.columns)
    if missing:
        raise ValueError(f"Fixture results missing columns: {sorted(missing)}")

    fx = fx[fx["Finished"].fillna(False)].copy()
    fx["FPLFixtureId"] = pd.to_numeric(fx["FPLFixtureId"], errors="coerce").astype("Int64")
    fx = fx.dropna(subset=["FPLFixtureId", "HomeGoals", "AwayGoals"])
    fx = fx[["FPLFixtureId", "HomeGoals", "AwayGoals"]].drop_duplicates("FPLFixtureId", keep="last")

    for col in [
        "ActualHomeGoals", "ActualAwayGoals", "ActualResult", "ActualOver2_5",
        "ActualBTTS", "SettledAtUTC",
    ]:
        if col not in out.columns:
            out[col] = pd.NA

    merged = out.merge(fx, on="FPLFixtureId", how="left", suffixes=("", "_official"))
    mask = merged["HomeGoals"].notna() & merged["AwayGoals"].notna()

    hg = pd.to_numeric(merged.loc[mask, "HomeGoals"], errors="coerce")
    ag = pd.to_numeric(merged.loc[mask, "AwayGoals"], errors="coerce")
    merged.loc[mask, "ActualHomeGoals"] = hg.to_numpy()
    merged.loc[mask, "ActualAwayGoals"] = ag.to_numpy()
    merged.loc[mask, "ActualResult"] = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    merged.loc[mask, "ActualOver2_5"] = ((hg + ag) >= 3).astype(int).to_numpy()
    merged.loc[mask, "ActualBTTS"] = ((hg > 0) & (ag > 0)).astype(int).to_numpy()

    needs_stamp = mask & merged["SettledAtUTC"].isna()
    merged.loc[needs_stamp, "SettledAtUTC"] = _utc_now_iso()

    return merged.drop(columns=["HomeGoals", "AwayGoals"], errors="ignore")


def multiclass_brier(df: pd.DataFrame) -> float:
    probs = df[["P_HomeWin", "P_Draw", "P_AwayWin"]].to_numpy(dtype=float)
    outcomes = np.zeros_like(probs)
    mapping = {"H": 0, "D": 1, "A": 2}
    for i, result in enumerate(df["ActualResult"].astype(str)):
        outcomes[i, mapping[result]] = 1.0
    return float(np.mean(np.sum((probs - outcomes) ** 2, axis=1)))


def multiclass_log_loss(df: pd.DataFrame) -> float:
    mapping = {"H": "P_HomeWin", "D": "P_Draw", "A": "P_AwayWin"}
    chosen = []
    for _, row in df.iterrows():
        chosen.append(float(row[mapping[str(row["ActualResult"])] ]))
    p = np.clip(np.asarray(chosen, dtype=float), 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def ranked_probability_score(df: pd.DataFrame) -> float:
    probs = df[["P_HomeWin", "P_Draw", "P_AwayWin"]].to_numpy(dtype=float)
    outcomes = np.zeros_like(probs)
    mapping = {"H": 0, "D": 1, "A": 2}
    for i, result in enumerate(df["ActualResult"].astype(str)):
        outcomes[i, mapping[result]] = 1.0
    cp = np.cumsum(probs, axis=1)[:, :-1]
    co = np.cumsum(outcomes, axis=1)[:, :-1]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2.0))


def binary_brier(prob: Iterable[float], outcome: Iterable[int]) -> float:
    p = np.asarray(list(prob), dtype=float)
    y = np.asarray(list(outcome), dtype=float)
    return float(np.mean((p - y) ** 2))


def binary_log_loss(prob: Iterable[float], outcome: Iterable[int]) -> float:
    p = np.clip(np.asarray(list(prob), dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(list(outcome), dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluation_summary(settled: pd.DataFrame) -> pd.DataFrame:
    df = settled.dropna(subset=["ActualResult", "ActualHomeGoals", "ActualAwayGoals"]).copy()
    if df.empty:
        return pd.DataFrame([{"Matches": 0}])

    probs = df[["P_HomeWin", "P_Draw", "P_AwayWin"]].to_numpy(dtype=float)
    predicted = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)]
    actual = df["ActualResult"].astype(str).to_numpy()

    over_col = "CalP_Over2_5" if "CalP_Over2_5" in df.columns else "RawP_Over2_5"
    metrics = {
        "Matches": int(len(df)),
        "1X2_Accuracy": float(np.mean(predicted == actual)),
        "1X2_Brier": multiclass_brier(df),
        "1X2_LogLoss": multiclass_log_loss(df),
        "1X2_RPS": ranked_probability_score(df),
        "BTTS_Brier": binary_brier(df["P_BTTS_Yes"], df["ActualBTTS"].astype(int)),
        "BTTS_LogLoss": binary_log_loss(df["P_BTTS_Yes"], df["ActualBTTS"].astype(int)),
        "Over2_5_Brier": binary_brier(df[over_col], df["ActualOver2_5"].astype(int)),
        "Over2_5_LogLoss": binary_log_loss(df[over_col], df["ActualOver2_5"].astype(int)),
        "HomeGoals_MAE": float(np.mean(np.abs(df["Lambda_Home_xG"].astype(float) - df["ActualHomeGoals"].astype(float)))),
        "AwayGoals_MAE": float(np.mean(np.abs(df["Lambda_Away_xG"].astype(float) - df["ActualAwayGoals"].astype(float)))),
        "TotalGoals_MAE": float(np.mean(np.abs(df["Lambda_Total_xG"].astype(float) - (df["ActualHomeGoals"].astype(float) + df["ActualAwayGoals"].astype(float))))),
    }
    return pd.DataFrame([metrics])


def calibration_table(settled: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    df = settled.dropna(subset=["ActualResult"]).copy()
    if df.empty:
        return pd.DataFrame()

    specs = [
        ("Home", "P_HomeWin", (df["ActualResult"] == "H").astype(int)),
        ("Draw", "P_Draw", (df["ActualResult"] == "D").astype(int)),
        ("Away", "P_AwayWin", (df["ActualResult"] == "A").astype(int)),
        ("BTTS", "P_BTTS_Yes", pd.to_numeric(df["ActualBTTS"], errors="coerce")),
    ]
    over_col = "CalP_Over2_5" if "CalP_Over2_5" in df.columns else "RawP_Over2_5"
    specs.append(("Over2.5", over_col, pd.to_numeric(df["ActualOver2_5"], errors="coerce")))

    rows: list[dict] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for market, prob_col, outcome in specs:
        temp = pd.DataFrame({"p": pd.to_numeric(df[prob_col], errors="coerce"), "y": outcome}).dropna()
        if temp.empty:
            continue
        temp["bucket"] = pd.cut(temp["p"], bins=edges, include_lowest=True, right=True)
        for bucket, group in temp.groupby("bucket", observed=True):
            rows.append({
                "Market": market,
                "ProbabilityBucket": str(bucket),
                "Count": int(len(group)),
                "MeanPredicted": float(group["p"].mean()),
                "ActualRate": float(group["y"].mean()),
                "CalibrationGap": float(group["p"].mean() - group["y"].mean()),
            })
    return pd.DataFrame(rows)


def rolling_evaluation(settled: pd.DataFrame, windows: tuple[int, ...] = (20, 50, 100)) -> pd.DataFrame:
    df = settled.dropna(subset=["ActualResult"]).copy()
    if df.empty:
        return pd.DataFrame()
    sort_cols = [c for c in ["Date", "KickoffUTC", "FPLFixtureId"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    rows = []
    for window in windows:
        if len(df) < window:
            continue
        sample = df.tail(window)
        summary = evaluation_summary(sample).iloc[0].to_dict()
        summary["Window"] = window
        rows.append(summary)
    return pd.DataFrame(rows)
