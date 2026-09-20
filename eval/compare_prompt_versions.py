import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def compare_versions():
    old = pd.read_csv(PROJECT_ROOT / "data" / "golden_eval_with_judge.csv")
    new = pd.read_csv(PROJECT_ROOT / "data" / "golden_eval_with_judge_v2.csv")

    old_pass = (old["judge_verdict"] == "PASS").mean()
    old_partial = (old["judge_criteria_met"] / old["judge_criteria_total"]).mean()
    new_pass = (new["judge_verdict"] == "PASS").mean()
    new_partial = (new["judge_criteria_met"] / new["judge_criteria_total"]).mean()

    print(f"OLD prompt (full 175) - PASS: {old_pass:.2%}, partial credit: {old_partial:.2%}")
    print(f"NEW prompt (full 175) - PASS: {new_pass:.2%}, partial credit: {new_partial:.2%}")

if __name__ == "__main__":
    compare_versions()