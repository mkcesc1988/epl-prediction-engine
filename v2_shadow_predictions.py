from __future__ import annotations

from pathlib import Path

import pandas as pd

from daily_predictions import build_daily_predictions
from pipeline import load_config


V2_VERSION = "V2-shadow.1"


def build_v2_shadow_predictions(cfg: dict) -> pd.DataFrame:
    """Create an isolated V2 shadow baseline without changing V1.2 production output.

    Phase 1 intentionally mirrors V1.2 probabilities. Future V2 context features
    (lineups/injuries, rest/congestion, recent/preseason form, richer strength data)
    should be added here, then evaluated head-to-head against the frozen V1.2 output.
    """
    shadow = build_daily_predictions(cfg).copy()
    if shadow.empty:
        return shadow

    shadow["BaselineModelVersion"] = shadow.get("ModelVersion", "V1.2")
    shadow["ModelVersion"] = V2_VERSION
    shadow["ShadowMode"] = True
    shadow["V2ContextAdjustmentApplied"] = False
    return shadow


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    shadow = build_v2_shadow_predictions(cfg)
    latest_path = out_dir / "v2_shadow_predictions_latest.csv"
    shadow.to_csv(latest_path, index=False)

    print(f"Saved V2 shadow predictions: {latest_path}")
    print("V1.2 production predictions and bet tracking were not modified.")
    if not shadow.empty:
        print(shadow.to_string(index=False))


if __name__ == "__main__":
    main()
