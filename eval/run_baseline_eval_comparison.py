import pandas as pd
from pathlib import Path
from eval.baselines import (
    trivial_baseline_fit, trivial_baseline_predict,
    nn_baseline_fit, nn_baseline_predict
)

from src.intents.classifier import load_few_shot_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_baseline_comparison():
    df = pd.read_csv(PROJECT_ROOT/"data"/"taxonomy_reading_sample_filled.csv")
    df = df.dropna(subset=["rough_bucket"])

    few_shot = load_few_shot_examples(example_per_label=1)
    few_shot_texts = {ex["text"] for ex in few_shot}
    eval_rows = df[~df["text_customer"].isin(few_shot_texts)]

    full_labeled =[
        {"text": row["text_customer"],"intent": row["rough_bucket"]}
        for _, row in df.iterrows()
    ]

    most_common_label = trivial_baseline_fit(full_labeled)
    nn_embeddings, nn_labels = nn_baseline_fit(few_shot)

    trivial_correct = 0
    nn_correct = 0

    for i,(_,row) in enumerate(eval_rows.iterrows(),start=1):
        true_label = row["rough_bucket"]
        message = row["text_customer"]

        trivial_pred = trivial_baseline_predict(message,most_common_label)
        nn_pred = nn_baseline_predict(message,nn_embeddings,nn_labels)

        if trivial_pred == true_label:
            trivial_correct+=1
        if nn_pred == true_label:
            nn_correct+=1

        print(f"[{i}/{len(eval_rows)}] true={true_label} trivial={trivial_pred} nn={nn_pred}")

    n = len(eval_rows)
    print(f"\nTrivial Baseline accuracy: {trivial_correct}/{n} = {trivial_correct/n: .2%}")
    print(f"NN Baseline accuracy: {nn_correct}/{n} = {nn_correct/n: .2%}")
    print(f"LLM classifier accuracy: 40/{n} = 80.00% (from earlier run)")

if __name__=="__main__":
    run_baseline_comparison()