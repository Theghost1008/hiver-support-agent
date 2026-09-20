"""
Inspects false negatives (should_escalate=Y, predicted=N) to find
real patterns before deciding whether/how to fix the escalation rules.
"""
import pandas as pd
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("data/golden_eval_with_escalation.csv")
false_negatives = df[(df["predicted_should_escalate"] == "N") & (df["should_escalate"] == "Y")]

print(f"Total false negatives: {len(false_negatives)}")
print(f"\nTrue intent distribution among false negatives:")
print(false_negatives["true_intent"].value_counts())

print("\n--- Sample of 8 false negatives ---")
sample = false_negatives.sample(min(8, len(false_negatives)), random_state=1)
for _, row in sample.iterrows():
    print(f"\nMessage: {row['text_customer']}")
    print(f"True intent: {row['true_intent']} | Predicted intent: {row['real_predicted_intent']}")
    print(f"Your escalation reason notes: {row['escalation_reason_notes']}")
    print(f"System's reason for NOT escalating: {row['escalation_reason']}")