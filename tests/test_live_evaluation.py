import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live_evaluation import (
    archive_predictions,
    calibration_table,
    evaluation_summary,
    settle_predictions,
)


class LiveEvaluationTests(unittest.TestCase):
    def _predictions(self):
        return pd.DataFrame([
            {
                "Season": "2026/27",
                "FPLFixtureId": 1,
                "ModelVersion": "V1.2",
                "Date": "2026-09-13",
                "HomeTeam": "A",
                "AwayTeam": "B",
                "P_HomeWin": 0.50,
                "P_Draw": 0.25,
                "P_AwayWin": 0.25,
                "P_BTTS_Yes": 0.60,
                "CalP_Over2_5": 0.55,
                "Lambda_Home_xG": 1.5,
                "Lambda_Away_xG": 1.0,
                "Lambda_Total_xG": 2.5,
            },
            {
                "Season": "2026/27",
                "FPLFixtureId": 2,
                "ModelVersion": "V1.2",
                "Date": "2026-09-13",
                "HomeTeam": "C",
                "AwayTeam": "D",
                "P_HomeWin": 0.20,
                "P_Draw": 0.30,
                "P_AwayWin": 0.50,
                "P_BTTS_Yes": 0.40,
                "CalP_Over2_5": 0.45,
                "Lambda_Home_xG": 0.8,
                "Lambda_Away_xG": 1.4,
                "Lambda_Total_xG": 2.2,
            },
        ])

    def _fixtures(self):
        return pd.DataFrame([
            {"FPLFixtureId": 1, "Finished": True, "HomeGoals": 2, "AwayGoals": 1},
            {"FPLFixtureId": 2, "Finished": True, "HomeGoals": 0, "AwayGoals": 1},
        ])

    def test_archive_is_first_seen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            first = self._predictions()
            archive_predictions(first, path)

            changed = first.copy()
            changed.loc[0, "P_HomeWin"] = 0.99
            saved = archive_predictions(changed, path)

            self.assertEqual(len(saved), 2)
            self.assertAlmostEqual(float(saved.loc[saved["FPLFixtureId"] == 1, "P_HomeWin"].iloc[0]), 0.50)

    def test_settlement_and_summary(self):
        settled = settle_predictions(self._predictions(), self._fixtures())
        self.assertEqual(settled["ActualResult"].tolist(), ["H", "A"])
        self.assertEqual(settled["ActualOver2_5"].astype(int).tolist(), [1, 0])
        self.assertEqual(settled["ActualBTTS"].astype(int).tolist(), [1, 0])

        summary = evaluation_summary(settled).iloc[0]
        self.assertEqual(int(summary["Matches"]), 2)
        self.assertAlmostEqual(float(summary["1X2_Accuracy"]), 1.0)
        self.assertGreater(float(summary["1X2_Brier"]), 0.0)
        self.assertGreater(float(summary["1X2_LogLoss"]), 0.0)

    def test_calibration_table(self):
        settled = settle_predictions(self._predictions(), self._fixtures())
        table = calibration_table(settled)
        self.assertFalse(table.empty)
        self.assertTrue({"Home", "Draw", "Away", "BTTS", "Over2.5"}.issubset(set(table["Market"])))


if __name__ == "__main__":
    unittest.main()
