import time
import pandas as pd
from pathlib import Path
from src.intents.classifier import classify_intent, load_few_shot_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_sanity_check():
    df=pd.read_csv(PROJECT_ROOT/"data"/"taxonomy_reading_sample_filled.csv")
    df=df.dropna(subset=["rough_bucket"])

    few_shot = load_few_shot_examples(example_per_label=1)
    few_shot_texts = {ex["text"] for ex in few_shot}
    eval_rows = df[~df["text_customer"].isin(few_shot_texts)]
    correct = 0
    mismatches = []

    for i,(_,row) in enumerate(eval_rows.iterrows(),start=1):
        true_label = row["rough_bucket"]
        predicted = classify_intent(row["text_customer"],few_shot)
        print(f"[{i}/{len(eval_rows)}] true={true_label} predicted={predicted}")
        if predicted == true_label:
            correct+=1
        else:
            mismatches.append({
                "text":row["text_customer"],
                "true":true_label,
                "predicted":predicted
            })
        time.sleep(2.5)
    accuracy = correct / len(eval_rows)
    print(f"Accuracy: {correct}/{len(eval_rows)} = {accuracy: .2%}")
    print(f"Mismatches")
    for m in mismatches:
        print(f" TRUE: {m['true']:20s} PRED: {m['predicted']:20s} | {m['text'][:60]}")

if __name__ == "__main__":
    run_sanity_check()