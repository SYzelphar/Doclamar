from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from schemas.state import DoclamarState
from schemas.models import DocumentSchema

logger = logging.getLogger(__name__)

STOPWORDS = {
    "explain", "tell", "me", "about", "what", "is", "the", "a",
    "an", "how", "does", "do", "give", "show", "find", "describe",
    "were", "was", "used", "using", "which", "type", "effect"
}

SUPPORTED_EXTS = {"pdf", "txt", "docx"}


def _broaden(query: str) -> List[str]:
    words = [w.lower() for w in query.split() if w.lower() not in STOPWORDS and len(w) > 2]
    return words if words else query.lower().split()[:4]


def _scan_all_files(root_path: str) -> List[Dict[str, Any]]:
    results = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            ext = Path(filename).suffix.lstrip(".").lower()
            if ext not in SUPPORTED_EXTS:
                continue
            full_path = os.path.join(dirpath, filename)
            size_kb = round(os.path.getsize(full_path) / 1024, 2)
            results.append(DocumentSchema(
                file_path=full_path,
                file_name=filename,
                file_type=ext,
                size_kb=size_kb,
            ).model_dump())
    return results


def replan_node(state: DoclamarState) -> DoclamarState:
    retry = state.get("retry_count", 0)
    logger.info(f"[ReplanNode] No results — broadening strategy (attempt {retry + 1})")

    original = state.get("routing_plan") or {}
    root_path = state.get("root_path", "")

    if retry >= 1 and root_path:
        logger.info("[ReplanNode] Retry 2 — scanning ALL supported files in directory unconditionally.")
        all_docs = _scan_all_files(root_path)
        new_plan = {
            **original,
            "keywords": _broaden(state["query"]),
            "allowed_extensions": list(SUPPORTED_EXTS),
            "max_files": 100,
            "use_content_preview": True,
            "reasoning": "Final fallback: all supported files included unconditionally.",
        }
        return {
            **state,
            "routing_plan": new_plan,
            "candidate_documents": all_docs,
            "parsed_chunks": None,
            "retrieved_chunks": None,
            "reranked_chunks": None,
            "retry_count": 1,
            "error": None,
        }

    new_plan = {
        **original,
        "keywords": _broaden(state["query"]),
        "allowed_extensions": list(SUPPORTED_EXTS),
        "max_files": original.get("max_files", 25) + 25,
        "use_content_preview": True,
        "reasoning": "Broadened after empty first-pass retrieval.",
    }

    return {
        **state,
        "routing_plan": new_plan,
        "candidate_documents": None,
        "parsed_chunks": None,
        "retrieved_chunks": None,
        "reranked_chunks": None,
        "retry_count": 1,
        "error": None,
    }
