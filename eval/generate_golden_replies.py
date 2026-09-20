import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from src.reply.generator import generate_reply
from src.retrieval.retriever import load_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_eval_with_real_predictions.csv"
# CHECKPOINT_PATH = PROJECT_ROOT / "data" / "golden_eval_with_replies.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "golden_eval_with_replies_v2.csv"

def generate_golden_replies():
    if CHECKPOINT_PATH.exists():
        print("Resuming from existing checkpoint")
        df = pd.read_csv(CHECKPOINT_PATH)
    else:
        df = pd.read_csv(GOLDEN_PATH)
        df["generated_reply"] = None
        df["num_grounded_examples"] = None
        df.to_csv(CHECKPOINT_PATH, index=False)

    already_done = df["generated_reply"].notna().sum()
    print(f"Already generated: {already_done}/{len(df)}")

    index = load_index()
    pending_indices = df[df["generated_reply"].isna()].index.tolist()

    progress_bar = tqdm(pending_indices, desc="Generating replies", initial=already_done, total=len(df))

    for count, idx in enumerate(progress_bar, start=1):
        message = str(df.at[idx, "text_customer"])
        result = generate_reply(message, index)

        if not result["reply"]:
            print(f"WARNING: empty reply generated for row {idx}: {message[:60]}")

        df.at[idx, "generated_reply"] = result["reply"]
        df.at[idx, "num_grounded_examples"] = result["num_grounded_examples"]

        if count % 10 == 0:
            df.to_csv(CHECKPOINT_PATH, index=False)

        time.sleep(2.5)

    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"\nDone. Saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    generate_golden_replies()