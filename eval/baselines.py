from collections import Counter
from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")

def trivial_baseline_fit(labeled_examples: list[dict])->str:
    labels = [ex["intent"] for ex in labeled_examples]
    most_common_label, _ = Counter(labels).most_common(1)[0]
    return most_common_label

def trivial_baseline_predict(message: str, most_common_label: str)->str:
    return most_common_label

def nn_baseline_fit(labeled_examples: list[dict]):
    texts = [ex["text"] for ex in labeled_examples]
    labels = [ex["intent"] for ex in labeled_examples]
    embeddings = _model.encode(texts,normalize_embeddings=True)
    return embeddings,labels

def nn_baseline_predict(message:str, fitted_embeddings, fitted_labels)->str:
    query_vec = _model.encode([message],normalize_embeddings=True)[0]
    similarities = fitted_embeddings @ query_vec
    best_idx = np.argmax(similarities)
    result = fitted_labels[best_idx]
    return result