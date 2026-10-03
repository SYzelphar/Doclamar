from __future__ import annotations
import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss

from embeddings.embedder import embed_texts
from fileindex.index_builder import _get_index_dir, INDEX_FILE, META_FILE

logger = logging.getLogger(__name__)

_index_cache: Optional[faiss.Index] = None
_meta_cache: Optional[List[Dict]] = None


def _load_index() -> Tuple[faiss.Index, List[Dict]]:
    global _index_cache, _meta_cache

    if _index_cache is not None and _meta_cache is not None:
        return _index_cache, _meta_cache

    index_dir = _get_index_dir()
    index_path = os.path.join(index_dir, INDEX_FILE)
    meta_path = os.path.join(index_dir, META_FILE)

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("File index not found. Please build it first.")

    _index_cache = faiss.read_index(index_path)
    with open(meta_path, "rb") as fh:
        _meta_cache = pickle.load(fh)

    logger.info(f"[FileIndexSearch] Loaded index: {len(_meta_cache)} files")
    return _index_cache, _meta_cache


def invalidate_cache():
    global _index_cache, _meta_cache
    _index_cache = None
    _meta_cache = None


def search_index(
    query: str,
    allowed_extensions: Optional[List[str]] = None,
    top_k: int = 20,
    score_threshold: float = 0.15,
) -> List[Dict]:
    index, metadata = _load_index()

    query_emb = embed_texts([query]).astype(np.float32)
    faiss.normalize_L2(query_emb)

    k = min(top_k * 3, len(metadata))
    scores, indices = index.search(query_emb, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if float(score) < score_threshold:
            continue

        file_info = metadata[idx]
        ext = file_info.get("file_type", "").lower()

        if allowed_extensions and ext not in allowed_extensions:
            continue

        if not os.path.exists(file_info["file_path"]):
            continue

        result = {**file_info, "index_score": round(float(score), 4)}
        results.append(result)

        if len(results) >= top_k:
            break

    logger.info(
        f"[FileIndexSearch] Query '{query[:40]}...' → {len(results)} file(s) matched"
    )
    return results
