"""Local ONNX models (via fastembed): a bi-encoder for search and a cross-encoder for reranking.

No PyTorch — this keeps the packaged app small. Models are loaded lazily and
cached in Settings.model_dir, so after the first download everything runs offline.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """Model could not be loaded (usually: first run without internet)."""


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return 384

    def _get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding
                    try:
                        self._model = TextEmbedding(
                            self.settings.embed_model, cache_dir=str(self.settings.model_dir)
                        )
                    except Exception as e:
                        raise ModelUnavailable(
                            f"Could not load embedding model '{self.settings.embed_model}'. "
                            "The first run needs internet access to download it."
                        ) from e
                    logger.info("Embedding model loaded: %s", self.settings.embed_model)
        return self._model

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.asarray(list(self._get().embed(list(texts), batch_size=batch_size)), dtype=np.float32)
        return _normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        vector = np.asarray(list(self._get().embed([self.settings.query_instruction + query])), dtype=np.float32)
        return _normalize(vector)[0]


class Reranker:
    """Cross-encoder that scores (query, passage) pairs. Optional: retrieval works without it."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._model is None and not self._failed:
            with self._lock:
                if self._model is None and not self._failed:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                    try:
                        self._model = TextCrossEncoder(
                            self.settings.rerank_model, cache_dir=str(self.settings.model_dir)
                        )
                        logger.info("Reranker loaded: %s", self.settings.rerank_model)
                    except Exception as e:
                        self._failed = True
                        logger.warning("Reranker unavailable, using fused ranking only: %s", e)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> Optional[List[float]]:
        model = self._load()
        if model is None or not passages:
            return None
        return [float(s) for s in model.rerank(query, list(passages))]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
