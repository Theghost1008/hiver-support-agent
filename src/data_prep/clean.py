import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "twcs_subsample.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def filter_apple_conversations(df: pd.DataFrame) -> pd.DataFrame:
    is_apple_reply = df["author_id"] == "AppleSupport"
    is_customer_to_apple = (df["inbound"] == True) & (
        df["text"].str.contains("@AppleSupport", case=False, na=False)
    )
    return df[is_apple_reply | is_customer_to_apple].copy()

def build_pairs(apple_df: pd.DataFrame) -> pd.DataFrame:
    customer_msgs = apple_df[apple_df["inbound"] == True]
    brand_replies = apple_df[apple_df["inbound"] == False]

    return customer_msgs.merge(
        brand_replies,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_apple")
    )

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    apple_df = filter_apple_conversations(df)
    pairs = build_pairs(apple_df)

    output_path = PROCESSED_DIR / "apple_pairs.csv"
    pairs[["text_customer", "text_apple"]].to_csv(output_path, index=False)
    print(f"Saved {len(pairs)} pairs to {output_path}")

    customer_msgs = apple_df[apple_df["inbound"] == True]
    brand_replies = apple_df[apple_df["inbound"] == False]
    print(f"Customer messages: {len(customer_msgs)}")
    print(f"Apple replies: {len(brand_replies)}")