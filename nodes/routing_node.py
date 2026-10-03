from __future__ import annotations
import os
import re
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from schemas.state import DoclamarState
from schemas.models import DocumentSchema
from utils.llm_provider import get_llm
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

STOPWORDS = {
    "explain", "tell", "me", "about", "what", "is", "the", "a",
    "an", "how", "does", "do", "give", "show", "find", "in", "of",
    "for", "describe", "summarize", "list", "all", "any", "some"
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _keyword_tokens(keywords: List[str]) -> List[str]:
    tokens = set()
    for kw in keywords:
        tokens.add(_normalize(kw))
        for word in _normalize(kw).split():
            if len(word) > 2:
                tokens.add(word)
    return list(tokens)


def _llm_routing_plan(query: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt = (
        f'User query: "{query}"\n\n'
        'Return a JSON routing plan:\n'
        '{\n'
        '  "keywords": ["keyword1", "keyword2"],\n'
        '  "allowed_extensions": ["pdf", "txt", "docx"],\n'
        '  "max_files": 25,\n'
        '  "use_content_preview": true,\n'
        '  "reasoning": "brief explanation"\n'
        '}\n\n'
        'Rules:\n'
        '- keywords: 3-6 core concepts, lowercase\n'
        '- allowed_extensions: only types likely to contain the answer\n'
        '- max_files: integer 10-50'
    )
    try:
        raw = llm.generate_json(prompt)
        data = extract_json(raw)
        assert isinstance(data.get("keywords"), list)
        assert isinstance(data.get("allowed_extensions"), list)
        return data
    except Exception as e:
        logger.warning(f"[RoutingNode] LLM plan failed: {e}. Using fallback.")
        keywords = [w for w in query.lower().split() if w not in STOPWORDS and len(w) > 2]
        return {
            "keywords": keywords or query.lower().split()[:4],
            "allowed_extensions": ["pdf", "txt", "docx"],
            "max_files": 25,
            "use_content_preview": True,
            "reasoning": "Fallback from query tokenization.",
        }


def _route_via_index(
    query: str,
    plan: Dict[str, Any],
    root_path: str,
) -> Tuple[List[DocumentSchema], Dict[str, Any], bool]:
    from fileindex.index_builder import index_exists
    from fileindex.index_search import search_index

    if not index_exists():
        return [], {}, False

    allowed_exts = [e.lower().lstrip(".") for e in plan.get("allowed_extensions", ["pdf", "txt", "docx"])]
    max_files = int(plan.get("max_files", 25))

    try:
        raw_results = search_index(
            query=query,
            allowed_extensions=allowed_exts,
            top_k=max_files,
            score_threshold=0.10,
        )

        # Filter to root_path if specified - STRICT FILTERING (no fallback)
        if root_path and root_path != ".":
            root_abs = os.path.abspath(root_path)
            raw_results = [
                r for r in raw_results
                if os.path.abspath(r["file_path"]).startswith(root_abs)
            ]
            # Removed the 'or raw_results' fallback - we enforce root_path strictly

        documents = [
            DocumentSchema(
                file_path=r["file_path"],
                file_name=r["file_name"],
                file_type=r["file_type"],
                size_kb=r.get("size_kb"),
            )
            for r in raw_results
            if os.path.exists(r["file_path"])
        ]

        search_stats = {
            "routing_method": "vector_index",
            "directories_scanned": len({os.path.dirname(r["file_path"]) for r in raw_results}),
            "directory_paths": sorted(list({os.path.dirname(r["file_path"]) for r in raw_results})),
            "total_files_seen": "indexed (not re-scanned)",
            "files_skipped_wrong_extension": 0,
            "files_skipped_not_relevant": 0,
            "files_matched": len(documents),
            "matched_file_names": [d.file_name for d in documents],
            "matched_file_paths": [d.file_path for d in documents],
            "keywords_used": plan.get("keywords", []),
            "extensions_searched": allowed_exts,
            "index_scores": [r.get("index_score", 0) for r in raw_results],
        }

        return documents, search_stats, True

    except Exception as e:
        logger.warning(f"[RoutingNode] Index search failed: {e}. Falling back to live scan.")
        return [], {}, False


def _route_via_scan(
    root_path: str,
    plan: Dict[str, Any],
) -> Tuple[List[DocumentSchema], Dict[str, Any]]:
    keywords = plan.get("keywords", [])
    allowed_exts = [e.lower().lstrip(".") for e in plan.get("allowed_extensions", ["pdf", "txt", "docx"])]
    max_files = int(plan.get("max_files", 25))
    use_preview = bool(plan.get("use_content_preview", True))
    kw_tokens = _keyword_tokens(keywords)

    results: List[DocumentSchema] = []
    dirs_scanned = set()
    total_files_seen = 0
    files_skipped_ext = 0
    files_skipped_relevance = 0

    for dirpath, _, filenames in os.walk(root_path):
        dirs_scanned.add(dirpath)
        for filename in filenames:
            if len(results) >= max_files:
                break

            total_files_seen += 1
            ext = Path(filename).suffix.lstrip(".").lower()

            if ext not in allowed_exts:
                files_skipped_ext += 1
                continue

            full_path = os.path.join(dirpath, filename)
            searchable = f"{full_path} {filename} {Path(filename).stem}"
            match = any(tok in _normalize(searchable) for tok in kw_tokens)

            if not match and use_preview:
                try:
                    with open(full_path, "rb") as fh:
                        preview = fh.read(2000).decode(errors="ignore")
                    match = any(tok in _normalize(preview) for tok in kw_tokens)
                except OSError:
                    pass

            if not match:
                files_skipped_relevance += 1
                continue

            size_kb = round(os.path.getsize(full_path) / 1024, 2)
            results.append(DocumentSchema(
                file_path=full_path,
                file_name=filename,
                file_type=ext,
                size_kb=size_kb,
            ))

    search_stats = {
        "routing_method": "live_scan",
        "directories_scanned": len(dirs_scanned),
        "directory_paths": sorted(list(dirs_scanned)),
        "total_files_seen": total_files_seen,
        "files_skipped_wrong_extension": files_skipped_ext,
        "files_skipped_not_relevant": files_skipped_relevance,
        "files_matched": len(results),
        "matched_file_names": [d.file_name for d in results],
        "matched_file_paths": [d.file_path for d in results],
        "keywords_used": kw_tokens,
        "extensions_searched": allowed_exts,
    }

    return results, search_stats


def routing_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[RoutingNode] Starting...")

    try:
        # Check if candidate_documents are already injected (user provided specific files)
        injected_docs = state.get("candidate_documents")
        if injected_docs:
            logger.info(
                f"[RoutingNode] Using {len(injected_docs)} pre-injected document(s). "
                f"Skipping routing search."
            )
            
            elapsed = round(time.time() - t0, 3)
            timings = dict(state.get("node_timings") or {})
            timings["routing"] = elapsed
            
            search_stats = {
                "routing_method": "injected",
                "directories_scanned": 0,
                "directory_paths": [],
                "total_files_seen": 0,
                "files_skipped_wrong_extension": 0,
                "files_skipped_not_relevant": 0,
                "files_matched": len(injected_docs),
                "matched_file_names": [d.get("file_name") for d in injected_docs],
                "matched_file_paths": [d.get("file_path") for d in injected_docs],
                "keywords_used": [],
                "extensions_searched": [],
                "note": "Using user-selected documents only. No filesystem search performed.",
            }
            
            return {
                **state,
                "routing_plan": {"source": "injected_documents"},
                "candidate_documents": injected_docs,
                "search_stats": search_stats,
                "error": None,
                "node_timings": timings,
            }
        
        # Otherwise, perform normal routing
        plan = _llm_routing_plan(state["query"])
        root_path = state["root_path"]

        documents, search_stats, used_index = _route_via_index(
            state["query"], plan, root_path
        )

        if not documents:
            if used_index:
                logger.info("[RoutingNode] Index returned no results. Falling back to live scan.")
            documents, search_stats = _route_via_scan(root_path, plan)

        elapsed = round(time.time() - t0, 3)
        timings = dict(state.get("node_timings") or {})
        timings["routing"] = elapsed

        method = search_stats.get("routing_method", "unknown")
        logger.info(
            f"[RoutingNode] [{method}] Matched {len(documents)} file(s) in {elapsed}s"
        )

        return {
            **state,
            "routing_plan": plan,
            "candidate_documents": [d.model_dump() for d in documents] if documents else None,
            "search_stats": search_stats,
            "error": None if documents else "No matching documents found.",
            "node_timings": timings,
        }

    except Exception as e:
        logger.error(f"[RoutingNode] Fatal: {e}")
        return {
            **state,
            "error": str(e),
            "candidate_documents": None,
            "search_stats": {},
            "node_timings": state.get("node_timings") or {},
        }
