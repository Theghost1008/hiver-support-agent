"""
Merges the blind human-labeled sample back with the hidden LLM judge
verdicts (using the saved indices), and computes real judge-human
agreement: raw percent agreement AND Cohen's kappa (chance-corrected).
"""
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def compute_agreement():
    full_judge_df = pd.read_csv(PROJECT_ROOT / "data" / "golden_eval_with_judge.csv")
    sample_indices = pd.read_csv(PROJECT_ROOT / "data" / "judge_agreement_sample_indices.csv")
    human_filled = pd.read_csv(PROJECT_ROOT / "data" / "judge_agreement_sample_filled.csv")

    idx_col = sample_indices.columns[0]
    original_rows = full_judge_df.loc[sample_indices[idx_col].tolist()].reset_index(drop=True)

    merged = human_filled.copy()
    merged["llm_verdict"] = original_rows["judge_verdict"].values
    merged["llm_criteria_met"] = original_rows["judge_criteria_met"].values
    merged["llm_criteria_total"] = original_rows["judge_criteria_total"].values

    raw_agreement = (merged["human_verdict"] == merged["llm_verdict"]).mean() # Raw percent agreement

    kappa = cohen_kappa_score(merged["human_verdict"], merged["llm_verdict"])  # Cohen's kappa: chance-corrected agreement

    print(f"Raw percent agreement: {raw_agreement:.2%}")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"\nHuman verdict distribution: {merged['human_verdict'].value_counts().to_dict()}")
    print(f"LLM verdict distribution: {merged['llm_verdict'].value_counts().to_dict()}")

    print("\nDisagreements:")
    disagreements = merged[merged["human_verdict"] != merged["llm_verdict"]]
    for _, row in disagreements.iterrows():
        print(f"  Human: {row['human_verdict']}, LLM: {row['llm_verdict']} | {row['text_customer'][:60]}")

    merged.to_csv(PROJECT_ROOT / "data" / "judge_agreement_final.csv", index=False)

if __name__ == "__main__":
    compute_agreement()