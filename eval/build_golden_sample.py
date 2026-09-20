import pandas as pd
from pathlib import Path
from src.intents.classifier import load_few_shot_examples
from src.intents.taxonomy import  INTENT_TAXONOMY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_INTENT_COUNT = 15
RANDOM_COUNT = 25

def build_golden_sample():
    presample = pd.read_csv(PROJECT_ROOT/"data"/"processed"/"presample_with_predictions.csv")
    full_pool = pd.read_csv(PROJECT_ROOT/"data"/"processed"/"apple_pairs.csv")
    few_shot = load_few_shot_examples()
    few_shot_texts = {ex["text"] for ex in few_shot}

    stratified_rows = []
    for intent in INTENT_TAXONOMY:
        subset = presample[presample["predicted_intent"]==intent]
        n_avail = len(subset)
        sampled = subset.sample(n=min(PER_INTENT_COUNT, n_avail),random_state=42)
        if n_avail < PER_INTENT_COUNT:
            print(f"WARNING: only {n_avail} examples available for '{intent}' (wanted {PER_INTENT_COUNT})")
        stratified_rows.append(sampled)

    stratified_df = pd.concat(stratified_rows)
    stratified_df["sample_source"] = "stratified"

    used_texts = set(stratified_df["text_customer"]) | few_shot_texts
    remaining_pool = full_pool[~full_pool["text_customer"].isin(used_texts)]
    random_sample = remaining_pool.sample(n=RANDOM_COUNT, random_state=99)
    random_sample["sample_source"] = "random"
    random_sample["predicted_intent"] = None

    golden_sample = pd.concat([
        stratified_df[["text_customer","text_apple","predicted_intent","sample_source"]],
        random_sample[["text_customer","text_apple","predicted_intent","sample_source"]],
    ]).reset_index(drop=True)
    output_path = PROJECT_ROOT/"data"/"golden_eval_unlabeled.csv"
    golden_sample.to_csv(output_path,index=False)

    print(f"\n Saved {len(golden_sample)} rows to {output_path}")
    print(golden_sample["sample_source"].value_counts())
    print("\nStratified breakdown by predicted intent:")
    print(stratified_df["predicted_intent"].value_counts())

if __name__ == "__main__":
    build_golden_sample()