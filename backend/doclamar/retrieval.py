"""Hybrid retrieval: vector + keyword search, fused with RRF, then cross-encoder reranking."""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    ocr: bool = False            # text was recognised from a scan or image

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
            if rerank_scores is not None:
                score, relevance = rerank_scores[n], _sigmoid(rerank_scores[n])
            else:
                score, relevance = fused[chunk_id], None
            hits.append(_hit(rows[chunk_id], score, relevance))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def overview(self, scope: Scope, limit: Optional[int] = None) -> List[Hit]:
        """Passages spread across the scope's documents, for "what is this about?" questions."""
        ids = self.store.overview_chunk_ids(scope, limit or self.settings.overview_chunks)
        return self._hits(ids)

    def pages(self, path: str, wanted: List[int], limit: int = 12) -> Tuple[List[Hit], dict]:
        """Passages on specific pages. `wanted` may count from the end: -1 is the last page.

        Returns (hits, info); when there is nothing to show, info["message"] says why
        (no such page, or the page has no readable text).
        """
        row = self.store.get_file(path)
        name = row["name"] if row else os.path.basename(path)
        total = row["num_pages"] if row else None
        info = {"file": name, "total": total, "pages": [], "message": None}
        if total is None:
            # Word, text and image files have no pages: only "first"/"last" make sense.
            if all(p < 0 for p in wanted) or all(p == 1 for p in wanted):
                return self._hits(self.store.edge_chunk_ids(path, from_end=wanted[0] < 0, limit=3)), info
            info["message"] = f"{name} has no page numbers (only PDFs and multi-page images do)."
            return [], info

        resolved = sorted({p if p > 0 else total + 1 + p for p in wanted})
        info["pages"] = [p for p in resolved if 1 <= p <= total]
        if not info["pages"]:
            asked = ", ".join(str(p) for p in resolved)
            info["message"] = f"{name} has {total} page{'s' if total != 1 else ''}, so there is no page {asked}."
            return [], info
        hits = self._hits(self.store.chunk_ids_on_pages(path, info["pages"], limit))
        if not hits:
            label = ", ".join(str(p) for p in info["pages"])
            info["message"] = (f"Page {label} of {name} has no readable text: it may be blank, a picture "
                               "without words, or handwriting OCR couldn't read.")
        return hits, info

    def _hits(self, ids: List[int]) -> List[Hit]:
        rows = self.store.get_chunks(ids)
        return [_hit(rows[i], 0.0, None) for i in ids if i in rows]


def _hit(row: dict, score: float, relevance: Optional[float]) -> Hit:
    return Hit(
        chunk_id=row["id"], path=row["path"], name=row["name"], text=row["text"],
        section=row["section"], page_start=row["page_start"], page_end=row["page_end"],
        score=round(score, 4), relevance=relevance,  # unrounded: thresholds go down to 1e-5
        ocr=bool(row["ocr"]),
    )
