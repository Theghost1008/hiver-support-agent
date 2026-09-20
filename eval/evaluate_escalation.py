import pandas as pd
from pathlib import Path
from src.routing.escalation import decide_escalation
from src.retrieval.retriever import load_index, retrieve_similar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_THRESHOLD = 0.5

def evaluate_escalation():
    df = pd.read_csv(PROJECT_ROOT / "data" / "golden_eval_with_real_predictions.csv")
    index = load_index()

    predicted_decisions = []
    predicted_reasons = []

    for _, row in df.iterrows():
        message = str(row["text_customer"])
        intent = row["real_predicted_intent"]

        retrieved = retrieve_similar(message, index, top_k=3)
        grounded = [ex for ex in retrieved if ex["similarity"] >= SIMILARITY_THRESHOLD]

        result = decide_escalation(message, intent, grounded)
        predicted_decisions.append(result["decision"])
        predicted_reasons.append(result["reason"])

    df["escalation_decision"] = predicted_decisions
    df["escalation_reason"] = predicted_reasons

    df["predicted_should_escalate"] = df["escalation_decision"].map({"escalate": "Y", "auto_handle": "N"})    # Normalize to Y/N for comparison against should_escalate

    accuracy = (df["predicted_should_escalate"] == df["should_escalate"]).mean()

    # Confusion matrix - the two error types have very different real-world costs
    true_positives = ((df["predicted_should_escalate"] == "Y") & (df["should_escalate"] == "Y")).sum()
    false_positives = ((df["predicted_should_escalate"] == "Y") & (df["should_escalate"] == "N")).sum()
    true_negatives = ((df["predicted_should_escalate"] == "N") & (df["should_escalate"] == "N")).sum()
    false_negatives = ((df["predicted_should_escalate"] == "N") & (df["should_escalate"] == "Y")).sum()

    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nConfusion matrix:")
    print(f"  True escalations correctly caught (TP):  {true_positives}")
    print(f"  Unnecessary escalations (FP):             {false_positives}")
    print(f"  Correctly auto-handled (TN):               {true_negatives}")
    print(f"  MISSED escalations - auto-handled real issues that needed a human (FN): {false_negatives}")

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"\nRecall on true escalations: {recall:.2%} (of messages that truly needed escalation, what fraction did we catch?)")
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision on escalations: {precision:.2%} (of messages we escalated, what fraction truly needed it?)")

    df.to_csv(PROJECT_ROOT / "data" / "golden_eval_with_escalation.csv", index=False)

if __name__ == "__main__":
    evaluate_escalation()