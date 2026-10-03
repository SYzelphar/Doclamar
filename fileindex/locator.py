from __future__ import annotations
import re
import os
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOCATE_PATTERNS = [
    r"\bwhere\s+(is|are|can i find|do i find)\b",
    r"\b(locate|find|search for|look for)\s+(the\s+)?(file|document|pdf|folder|directory)\b",
    r"\bcan('t| not)?\s+(you\s+)?(find|locate|tell me where)\b",
    r"\b(not able to|unable to)\s+(find|locate)\b",
    r"\bwhere.*\b(file|document|pdf|stored|saved|located)\b",
    r"\b(path|location|directory)\s+(of|for|to)\b",
    r"\bshow me\s+(where|the path|the location)\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in LOCATE_PATTERNS]


def is_locate_query(query: str) -> bool:
    return any(p.search(query) for p in _compiled)


def locate_files(
    query: str,
    top_k: int = 10,
    allowed_extensions: Optional[List[str]] = None,
) -> List[Dict]:
    from fileindex.index_builder import index_exists
    from fileindex.index_search import search_index

    if not index_exists():
        return []

    results = search_index(
        query=query,
        allowed_extensions=allowed_extensions,
        top_k=top_k,
        score_threshold=0.08,
    )

    return results


def format_locate_results(results: List[Dict], query: str) -> str:
    if not results:
        return (
            "I searched the file index but could not find any files matching your query.\n"
            "Make sure your file index is up to date by running: python main.py index"
        )

    lines = [f"I found {len(results)} file(s) that may match your query:\n"]
    for i, r in enumerate(results, 1):
        name = r.get("file_name", "unknown")
        path = r.get("file_path", "unknown")
        size = r.get("size_kb", 0)
        score = r.get("index_score", 0)
        lines.append(
            f"  [{i}] {name}\n"
            f"       Path  : {path}\n"
            f"       Size  : {size} KB\n"
            f"       Match : {round(score * 100, 1)}% relevance\n"
        )

    return "\n".join(lines)
