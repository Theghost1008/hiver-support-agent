import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.intents.classifier import classify_intent_batch, load_few_shot_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRE_SAMPLE_SIZE = 1500
BATCH_SIZE = 15
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "presample_with_predictions.csv"

def classify_presample():
    pairs = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "apple_pairs.csv")

    if CHECKPOINT_PATH.exists():
        print("Resuming from existing checkpoint")
        presaample = pd.read_csv(CHECKPOINT_PATH)
        already_done = presaample["predicted_intent"].notna().sum()
        print(f"Already classified: {already_done}/{len(presaample)}")
    else:
        presaample = pairs.sample(n=PRE_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        presaample["predicted_intent"] = None
        presaample.to_csv(CHECKPOINT_PATH, index=False)

    few_shot = load_few_shot_examples()

    pending_indices = presaample[presaample["predicted_intent"].isna()].index.tolist()

    batches = [pending_indices[i:i + BATCH_SIZE] for i in range(0, len(pending_indices), BATCH_SIZE)]

    for batch_num, batch_indices in enumerate(tqdm(batches, desc="Classifying batches"), start=1):
        batch_messages = presaample.loc[batch_indices, "text_customer"].tolist()
        labels = classify_intent_batch(batch_messages, few_shot)

        for idx, label in zip(batch_indices, labels):
            presaample.at[idx, "predicted_intent"] = label

        presaample.to_csv(CHECKPOINT_PATH, index=False)

    print(f"\nDone. All {len(presaample)} rows classified and saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    classify_presample()