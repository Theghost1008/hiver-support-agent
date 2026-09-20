import pandas as pd
from pathlib import Path
from eval.baselines import trivial_baseline_fit, trivial_baseline_predict, nn_baseline_fit, nn_baseline_predict
from src.intents.classifier import load_few_shot_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_final_baseline_comparison():
    golden = pd.read_csv(PROJECT_ROOT / "data" / "golden_eval_with_real_predictions.csv")
    full_pairs = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "apple_pairs.csv")

    full_labeled_proxy = [{"text": t, "intent": i} for t, i in zip(golden["text_customer"], golden["true_intent"])]
    most_common_label = trivial_baseline_fit(full_labeled_proxy)

    few_shot = load_few_shot_examples()
    nn_embeddings, nn_labels = nn_baseline_fit(few_shot)

    trivial_preds = [trivial_baseline_predict(msg, most_common_label) for msg in golden["text_customer"]]
    nn_preds = [nn_baseline_predict(msg, nn_embeddings, nn_labels) for msg in golden["text_customer"]]

    golden["trivial_pred"] = trivial_preds
    golden["nn_pred"] = nn_preds

    trivial_acc = (golden["trivial_pred"] == golden["true_intent"]).mean()
    nn_acc = (golden["nn_pred"] == golden["true_intent"]).mean()
    llm_acc = (golden["real_predicted_intent"] == golden["true_intent"]).mean()

    print(f"Trivial baseline accuracy: {trivial_acc:.2%}")
    print(f"NN baseline accuracy:      {nn_acc:.2%}")
    print(f"LLM classifier accuracy:   {llm_acc:.2%}")

    golden.to_csv(PROJECT_ROOT / "data" / "golden_eval_final_comparison.csv", index=False)

if __name__ == "__main__":
    run_final_baseline_comparison()