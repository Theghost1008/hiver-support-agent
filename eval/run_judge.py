import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from eval.llm_judge import judge_reply

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_eval_with_replies.csv"
# CHECKPOINT_PATH = PROJECT_ROOT / "data" / "golden_eval_with_judge.csv"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_eval_with_replies_v2.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "golden_eval_with_judge_v2.csv"

def run_judge():
    if CHECKPOINT_PATH.exists():
        print("Resuming from existing checkpoint")
        df = pd.read_csv(CHECKPOINT_PATH)
    else:
        df = pd.read_csv(GOLDEN_PATH)
        df["judge_verdict"] = None
        df["judge_criteria_met"] = None
        df["judge_criteria_total"] = None
        df["judge_reason"] = None
        df.to_csv(CHECKPOINT_PATH, index=False)

    already_done = df["judge_verdict"].notna().sum()
    print(f"Already judged: {already_done}/{len(df)}")

    pending_indices = df[df["judge_verdict"].isna()].index.tolist()
    progress_bar = tqdm(pending_indices, desc="Judging replies", initial=already_done, total=len(df))

    for count, idx in enumerate(progress_bar, start=1):
        result = judge_reply(
            message=str(df.at[idx, "text_customer"]),
            criteria=str(df.at[idx, "good_reply_criteria"]),
            reply=str(df.at[idx, "generated_reply"]),
        )
        df.at[idx, "judge_verdict"] = result["verdict"]
        df.at[idx, "judge_criteria_met"] = result["criteria_met"]
        df.at[idx, "judge_criteria_total"] = result["criteria_total"]
        df.at[idx, "judge_reason"] = result["reason"]

        if count % 10 == 0:
            df.to_csv(CHECKPOINT_PATH, index=False)
        time.sleep(2.5)

    df.to_csv(CHECKPOINT_PATH, index=False)
    pass_rate = (df["judge_verdict"] == "PASS").mean()
    partial_credit_rate = (df["judge_criteria_met"] / df["judge_criteria_total"]).mean()
    print(f"\nStrict PASS rate: {pass_rate:.2%}")
    print(f"Average partial credit (criteria met / total): {partial_credit_rate:.2%}")

if __name__ == "__main__":
    run_judge()