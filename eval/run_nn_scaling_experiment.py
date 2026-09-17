import pandas as pd
from pathlib import Path
from eval.baselines import nn_baseline_fit, nn_baseline_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_nn_data_scaling_experiment():
    df = pd.read_csv(PROJECT_ROOT / "data" / "taxonomy_reading_sample_filled.csv")
    df = df.dropna(subset=["rough_bucket"])

    eval_set = df.sample(n=15, random_state=7)
    fit_set = df.drop(eval_set.index)

    fit_examples = [
        {"text": row["text_customer"], "intent": row["rough_bucket"]}
        for _, row in fit_set.iterrows()
    ]
    embeddings, labels = nn_baseline_fit(fit_examples)

    correct = 0
    for i, (_, row) in enumerate(eval_set.iterrows(), start=1):
        true_label = row["rough_bucket"]
        predicted = nn_baseline_predict(row["text_customer"], embeddings, labels)
        print(f"[{i}/15] true={true_label} predicted={predicted}")
        if predicted == true_label:
            correct += 1

    print(f"\nNN baseline accuracy (fit on 45 examples): {correct}/15 = {correct/15:.2%}")

if __name__ == "__main__":
    run_nn_data_scaling_experiment()