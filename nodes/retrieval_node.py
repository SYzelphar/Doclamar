from __future__ import annotations
import time
import logging
from typing import Dict, List, Tuple

from schemas.state import DoclamarState
from schemas.models import ParsedChunkSchema
from vectorstore.faiss_store import FAISSStore
from utils.llm_provider import get_llm
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)


def _expand_query(query: str) -> List[str]:
    llm = get_llm()
    prompt = (
        f'Original query: "{query}"\n\n'
        'Generate 3 alternative search queries that would retrieve the same information '
        'using different phrasing, synonyms, or related terms.\n\n'
        'Return JSON: {"queries": ["variant1", "variant2", "variant3"]}\n\n'
        'Keep each variant concise (under 15 words). Do not explain.'
    )
    try:
        raw = llm.generate_json(prompt)
        data = extract_json(raw)
        variants = data.get("queries", [])
        if isinstance(variants, list) and variants:
            all_queries = [query] + [str(v) for v in variants[:3]]
            logger.info(f"[RetrievalNode] Query expanded to {len(all_queries)} variants")
            return all_queries
    except Exception as e:
        logger.warning(f"[RetrievalNode] Query expansion failed: {e}. Using original.")
    return [query]


def _deduplicate(
    results: List[Tuple[ParsedChunkSchema, float]],
    top_k: int,
) -> List[Tuple[ParsedChunkSchema, float]]:
    seen = set()
    deduped = []
    for chunk, score in results:
        key = chunk.metadata.get("chunk_index", "") + chunk.metadata.get("file_path", "")
        if key not in seen:
            seen.add(key)
            deduped.append((chunk, score))
        if len(deduped) >= top_k:
            break
    return deduped


def retrieval_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[RetrievalNode] Building FAISS index and searching...")

    raw_chunks = state.get("parsed_chunks") or []
    if not raw_chunks:
        return {**state, "retrieved_chunks": [], "error": "No parsed chunks to index."}

    chunks = [ParsedChunkSchema(**c) for c in raw_chunks]
    query  = state["query"]
    top_k  = state.get("top_k", 5)

    retrieve_k = max(top_k * 4, 20)

    try:
        store = FAISSStore()
        store.build(chunks)

        query_variants = _expand_query(query)

        all_results: List[Tuple[ParsedChunkSchema, float]] = []
        seen_keys = set()

        for variant in query_variants:
            variant_results = store.search(variant, top_k=retrieve_k)
            for chunk, score in variant_results:
                key = chunk.metadata.get("chunk_index", "") + chunk.metadata.get("file_path", "")
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append((chunk, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        final = _deduplicate(all_results, top_k=retrieve_k)

        retrieved = []
        for chunk, score in final:
            d = chunk.model_dump()
            d["score"] = score
            retrieved.append(d)

        elapsed = round(time.time() - t0, 3)
        timings = dict(state.get("node_timings") or {})
        timings["retrieval"] = elapsed

        logger.info(
            f"[RetrievalNode] {len(query_variants)} query variant(s) → "
            f"{len(retrieved)} unique chunk(s) in {elapsed}s"
        )
        return {
            **state,
            "retrieved_chunks": retrieved,
            "error": None,
            "node_timings": timings,
        }
    except Exception as e:
        logger.error(f"[RetrievalNode] Error: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}
