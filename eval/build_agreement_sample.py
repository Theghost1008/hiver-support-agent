import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SIZE = 30

def build_agreement_sample():
    df = pd.read_csv(PROJECT_ROOT / "data" / "golden_eval_with_judge.csv")
    sample = df.sample(n=SAMPLE_SIZE, random_state=123)

    blind_sample = sample[["text_customer", "good_reply_criteria", "generated_reply"]].copy()
    blind_sample["human_verdict"] = ""       # to be filled: PASS or FAIL
    blind_sample["human_criteria_met"] = ""  # to be filled: number
    blind_sample["human_criteria_total"] = ""  # to be filled: number

    output_path = PROJECT_ROOT / "data" / "judge_agreement_sample_blind.csv"
    blind_sample.to_csv(output_path, index=False)
    
    sample.index.to_series().to_csv(PROJECT_ROOT / "data" / "judge_agreement_sample_indices.csv", index=False)

    print(f"Saved {len(blind_sample)} rows to {output_path} for blind manual scoring.")

if __name__ == "__main__":
    build_agreement_sample()