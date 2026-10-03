from __future__ import annotations
import time
import math
import logging
import re
from collections import Counter
from typing import List, Tuple

import numpy as np

from schemas.state import DoclamarState
from schemas.models import ParsedChunkSchema
from embeddings.embedder import embed_texts
from utils.llm_provider import get_llm
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

MIN_SCORE_THRESHOLD = 0.0
ALPHA = 0.6
BETA = 0.4


def _cosine_scores(query: str, chunks: List[ParsedChunkSchema]) -> List[float]:
    texts = [c.content for c in chunks]
    all_texts = [query] + texts
    embeddings = embed_texts(all_texts)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normalized = embeddings / norms
    q_vec = normalized[0]
    c_vecs = normalized[1:]
    return (c_vecs @ q_vec).tolist()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_scores(query: str, chunks: List[ParsedChunkSchema], k1: float = 1.5, b: float = 0.75) -> List[float]:
    tokenized_docs = [_tokenize(c.content) for c in chunks]
    query_terms = _tokenize(query)
    N = len(tokenized_docs)
    avg_dl = sum(len(d) for d in tokenized_docs) / max(N, 1)

    df = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] += 1

    scores = []
    for doc_tokens in tokenized_docs:
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_score = (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_score
        scores.append(score)

    max_s = max(scores) if scores else 1.0
    return [s / max_s if max_s > 0 else 0.0 for s in scores]


def _llm_judge(query: str, chunks: List[ParsedChunkSchema], top_n: int = 10) -> List[float]:
    llm = get_llm()
    actual_n = min(top_n, len(chunks))
    candidates = chunks[:actual_n]

    numbered = "\n\n".join(
        f"[{i}] {c.content[:250]}" for i, c in enumerate(candidates)
    )

    prompt = (
        f'Query: "{query}"\n\n'
        f"Rate each of the {actual_n} chunks below for relevance (0.0 to 1.0).\n"
        f"You MUST return exactly {actual_n} scores.\n\n"
        f"{numbered}\n\n"
        f'Return ONLY: {{"scores": [s0, s1, ..., s{actual_n - 1}]}}'
    )

    try:
        raw = llm.generate_json(prompt)
        data = extract_json(raw)
        scores = data.get("scores", [])

        if not isinstance(scores, list):
            raise ValueError("scores is not a list")

        scores = [float(s) for s in scores]

        if len(scores) > actual_n:
            scores = scores[:actual_n]
        elif len(scores) < actual_n:
            scores = scores + [0.5] * (actual_n - len(scores))

        padded = scores + [0.0] * (len(chunks) - actual_n)
        return padded

    except Exception as e:
        logger.warning(f"[RerankNode] LLM judge failed: {e}. Using neutral scores.")
        return [0.5] * len(chunks)


def reranking_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[RerankingNode] Hybrid reranking...")

    raw = state.get("retrieved_chunks") or []
    if not raw:
        return {**state, "reranked_chunks": [], "retry_count": 1}

    chunks = [ParsedChunkSchema(**{k: v for k, v in c.items() if k != "score"}) for c in raw]
    query = state["query"]
    top_k = state.get("top_k", 5)

    try:
        cosine  = _cosine_scores(query, chunks)
        bm25    = _bm25_scores(query, chunks)
        llm_sc  = _llm_judge(query, chunks, top_n=min(10, len(chunks)))

        hybrid = [
            ALPHA * c + BETA * b + 0.1 * l
            for c, b, l in zip(cosine, bm25, llm_sc)
        ]

        scored = sorted(zip(hybrid, chunks), key=lambda x: x[0], reverse=True)
        scored = [(s, c) for s, c in scored if s >= MIN_SCORE_THRESHOLD]

        reranked = []
        for score, chunk in scored[:top_k]:
            d = chunk.model_dump()
            d["score"] = round(score, 4)
            reranked.append(d)

        elapsed = round(time.time() - t0, 3)
        timings = dict(state.get("node_timings") or {})
        timings["reranking"] = elapsed

        logger.info(f"[RerankingNode] {len(reranked)} chunk(s) after reranking in {elapsed}s")
        return {
            **state,
            "reranked_chunks": reranked,
            "error": None,
            "node_timings": timings,
        }
    except Exception as e:
        logger.error(f"[RerankingNode] Error: {e}")
        return {**state, "reranked_chunks": [], "error": str(e)}
