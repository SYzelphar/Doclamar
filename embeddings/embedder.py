from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

_model_instance: SentenceTransformer | None = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(MODEL_NAME)
    return _model_instance


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)
