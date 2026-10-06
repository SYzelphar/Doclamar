"""Wires the pieces together. One Engine per process (API, CLI or eval)."""
from __future__ import annotations

from typing import List, Optional

from .config import Settings
from .embeddings import Embedder, Reranker
from .history import ChatHistory
from .indexer import Indexer
from .ocr import OcrEngine
from .llm import LLMClient
from .pipeline import build_pipeline
from .retrieval import Retriever
from .store import IndexStore, Scope


class Engine:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[LLMClient] = None,
        ocr: Optional[OcrEngine] = None,
        use_reranker: bool = True,
    ):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.store = IndexStore(self.settings.index_db)
        self.embedder = embedder or Embedder(self.settings)
        if reranker is None and use_reranker:
            reranker = Reranker(self.settings)
        self.reranker = reranker
        self.ocr = ocr if ocr is not None else (OcrEngine() if self.settings.ocr_enabled else None)
        self.indexer = Indexer(self.store, self.embedder, self.settings, ocr=self.ocr)
        self.retriever = Retriever(self.store, self.embedder, self.reranker, self.settings)
        self.history = ChatHistory(self.settings.chats_db)
        self.llm = llm or LLMClient.from_env()
        self.pipeline = build_pipeline(self.retriever, self.llm, self.settings)

    def answer(self, question: str, scope: Scope, history: Optional[List[dict]] = None) -> dict:
        state = self.pipeline.invoke({"question": question, "scope": scope, "history": history or []})
        return {
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "status": state.get("status", "answered"),
            "search_query": state.get("search_query", question),
            "timings": state.get("timings", {}),
        }

    def close(self) -> None:
        self.indexer.shutdown()
