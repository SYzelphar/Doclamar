"""Hybrid retrieval: vector + keyword search, fused with RRF, then cross-encoder reranking."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from .config import Settings
from .embeddings import Embedder, Reranker
from .store import IndexStore, Scope

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    path: str
    name: str
    text: str
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    score: float                 # what the list is sorted by
    relevance: Optional[float]   # 0..1 from the cross-encoder; None without a reranker

    @property
    def pages(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"pp. {self.page_start}–{self.page_end}"
        return f"p. {self.page_start}"

    def to_dict(self) -> dict:
        return asdict(self)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> Dict[int, float]:
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Retriever:
    def __init__(self, store: IndexStore, embedder: Embedder, reranker: Optional[Reranker], settings: Settings):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings

    def search(self, query: str, scope: Scope, top_k: Optional[int] = None) -> List[Hit]:
        s = self.settings
        top_k = top_k or s.top_k

        vector_hits = self.store.vector_search(self.embedder.embed_query(query), scope, s.candidate_pool)
        keyword_hits = self.store.keyword_search(query, scope, s.candidate_pool)
        fused = reciprocal_rank_fusion([[i for i, _ in vector_hits], [i for i, _ in keyword_hits]])
        if not fused:
            return []

        pool = sorted(fused, key=fused.get, reverse=True)[: s.rerank_pool]
        rows = self.store.get_chunks(pool)
        pool = [i for i in pool if i in rows]

        rerank_scores = None
        if self.reranker is not None:
            passages = [
                f"{rows[i]['section']}\n{rows[i]['text']}" if rows[i]["section"] else rows[i]["text"]
                for i in pool
            ]
            rerank_scores = self.reranker.score(query, passages)

        hits = []
        for n, chunk_id in enumerate(pool):
            row = rows[chunk_id]
            if rerank_scores is not None:
                score, relevance = rerank_scores[n], _sigmoid(rerank_scores[n])
            else:
                score, relevance = fused[chunk_id], None
            hits.append(Hit(
                chunk_id=chunk_id, path=row["path"], name=row["name"], text=row["text"],
                section=row["section"], page_start=row["page_start"], page_end=row["page_end"],
                score=round(score, 4), relevance=None if relevance is None else round(relevance, 4),
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
