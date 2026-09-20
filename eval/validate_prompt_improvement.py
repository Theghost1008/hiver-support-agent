import pandas as pd
from tqdm import tqdm
import time
from src.reply.generator import generate_reply
from src.retrieval.retriever import load_index
from eval.llm_judge import judge_reply

def validate_prompt_improvement(sample_size=20):
    df = pd.read_csv("data/golden_eval_with_judge.csv")
    sample = df.sample(n=sample_size, random_state=77)
    index = load_index()

    old_pass = (sample["judge_verdict"] == "PASS").mean()
    old_partial = (sample["judge_criteria_met"] / sample["judge_criteria_total"]).mean()

    new_verdicts = []
    new_partials = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Testing updated prompt"):
        result = generate_reply(str(row["text_customer"]), index)
        judged = judge_reply(str(row["text_customer"]), str(row["good_reply_criteria"]), result["reply"])
        new_verdicts.append(judged["verdict"])
        new_partials.append(judged["criteria_met"] / max(judged["criteria_total"], 1))
        time.sleep(2.5)

    new_pass = (pd.Series(new_verdicts) == "PASS").mean()
    new_partial = pd.Series(new_partials).mean()

    print(f"\nOLD prompt - PASS rate: {old_pass:.2%}, partial credit: {old_partial:.2%}")
    print(f"NEW prompt - PASS rate: {new_pass:.2%}, partial credit: {new_partial:.2%}")

if __name__ == "__main__":
    validate_prompt_improvement()