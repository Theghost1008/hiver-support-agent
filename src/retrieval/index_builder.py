import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_retrieval_index(pairs_df: pd.DataFrame)->dict:
    texts = pairs_df["text_customer"].tolist()
    embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return {
        "embeddings":embeddings,
        "customer_texts":pairs_df["text_customer"].tolist(),
        "apple_replies":pairs_df["text_apple"].tolist()
    }

def save_index(index:dict, path: Path):
    np.savez(
        path,
        embeddings=index["embeddings"],
        customer_texts = np.array(index["customer_texts"], dtype=object),
        apple_replies = np.array(index["apple_replies"], dtype=object)
    )

if __name__ == "__main__":
    pairs_df = pd.read_csv(PROJECT_ROOT/"data"/"processed"/"apple_pairs.csv")
    index = build_retrieval_index(pairs_df)
    save_index(index,PROJECT_ROOT/"data"/"processed"/"retrieval_index.npz")
    print(f"Index built and saved: {len(index['customer_texts'])} historical pairs")