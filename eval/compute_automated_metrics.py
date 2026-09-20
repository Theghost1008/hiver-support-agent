import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def contains_link(text: str) -> bool:
    return bool(re.search(r'https?://\S+', text))

def compute_automated_metrics():
    df = pd.read_csv(PROJECT_ROOT / "data" / "golden_eval_with_replies.csv")

    replies = df["generated_reply"].fillna("")

    df["is_empty"] = replies.str.strip() == ""
    df["reply_length"] = replies.str.len()
    df["contains_link"] = replies.apply(contains_link)
    df["is_too_long"] = df["reply_length"] > 280

    print(f"Total replies: {len(df)}")
    print(f"Empty replies: {df['is_empty'].sum()} ({df['is_empty'].mean():.1%})")
    print(f"Replies containing a link: {df['contains_link'].sum()} ({df['contains_link'].mean():.1%})")
    print(f"Replies over 280 chars: {df['is_too_long'].sum()} ({df['is_too_long'].mean():.1%})")
    print(f"Average reply length: {df['reply_length'].mean():.0f} characters")
    print(f"\nGrounding: avg {df['num_grounded_examples'].mean():.2f} examples per reply")
    print(f"Replies with zero grounding: {(df['num_grounded_examples'] == 0).sum()} ({(df['num_grounded_examples'] == 0).mean():.1%})")

    df.to_csv(PROJECT_ROOT / "data" / "golden_eval_with_auto_metrics.csv", index=False)

if __name__ == "__main__":
    compute_automated_metrics()