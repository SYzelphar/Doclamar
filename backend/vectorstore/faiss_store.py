from __future__ import annotations
import numpy as np
import faiss
from typing import List, Tuple
from schemas.models import ParsedChunkSchema
from embeddings.embedder import embed_texts


class FAISSStore:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunks: List[ParsedChunkSchema] = []

    def build(self, chunks: List[ParsedChunkSchema]):
        if not chunks:
            return
        texts = [c.content for c in chunks]
        embeddings = embed_texts(texts)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query: str, top_k: int) -> List[Tuple[ParsedChunkSchema, float]]:
        if self.index is None or not self.chunks:
            return []
        q_emb = embed_texts([query])
        faiss.normalize_L2(q_emb)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_emb, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results
