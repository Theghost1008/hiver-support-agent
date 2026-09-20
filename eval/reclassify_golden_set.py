import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from src.intents.classifier import classify_intent, load_few_shot_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_eval_labeled.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "golden_eval_with_real_predictions.csv"

def reclassify_golden_set():
    if CHECKPOINT_PATH.exists():
        print("Resuming from existing checkpoint")
        df = pd.read_csv(CHECKPOINT_PATH)
    else:
        df = pd.read_csv(GOLDEN_PATH)
        df["real_predicted_intent"] = None
        df.to_csv(CHECKPOINT_PATH, index=False)

    already_done = df["real_predicted_intent"].notna().sum()
    print(f"Already classified: {already_done}/{len(df)}")

    few_shot = load_few_shot_examples()
    pending_indices = df[df["real_predicted_intent"].isna()].index.tolist()

    progress_bar = tqdm(pending_indices, desc="Re-classifying golden set", initial=already_done, total=len(df))

    for count, idx in enumerate(progress_bar, start=1):
        message = str(df.at[idx, "text_customer"])
        predicted = classify_intent(message, few_shot)
        df.at[idx, "real_predicted_intent"] = predicted

        if count % 10 == 0:
            df.to_csv(CHECKPOINT_PATH, index=False)

        time.sleep(2.5)

    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"\nDone. Saved to {CHECKPOINT_PATH}")

    accuracy = (df["real_predicted_intent"] == df["true_intent"]).mean()
    print(f"\nAccuracy (real classify_intent vs true_intent, all 175 rows): {accuracy:.2%}")

if __name__ == "__main__":
    reclassify_golden_set()