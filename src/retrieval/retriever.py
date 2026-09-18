import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT/"data"/"processed"/"retrieval_index.npz"

_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index(path: Path = INDEX_PATH)->dict:
    data = np.load(path,allow_pickle=True)
    return {
        "embeddings":data["embeddings"],
        "customer_texts":data["customer_texts"],
        "apple_replies":data["apple_replies"],
    }

def retrieve_similar(message:str, index:dict, top_k:int=3)->list[dict]:
    query_vec = _model.encode([message],normalize_embeddings=True)[0]
    similarities = index["embeddings"] @ query_vec
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        {
            "similarity":float(similarities[i]),
            "historical_customer_texts":index["customer_texts"][i],
            "historical_apple_reply":index["apple_replies"][i],
        }
        for i in top_indices
    ]

if __name__=="__main__":
    index = load_index()
    test_msg = "my phone screen went completely black and won't respond to anything"
    # query_vec = _model.encode([test_msg], normalize_embeddings=True)[0]
    # debug_similarities = index["embeddings"] @ query_vec
    # print("DEBUG top_indices:", np.argsort(debug_similarities)[::-1][:3])
    # print("DEBUG unique similarity values in whole array:", len(np.unique(debug_similarities)))
    # print("DEBUG total similarities:", len(debug_similarities))
    results = retrieve_similar(test_msg, index,top_k=3)
    print(f"Query: {test_msg}\n")
    for r in results:
        print(f"Similarity: {r['similarity']:.6f}")
        print(f"    Historical customer message: {r['historical_customer_texts']}")
        print(f"    Apple's actual reply: {r['historical_apple_reply']}")
        print()
