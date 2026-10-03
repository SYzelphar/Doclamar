from __future__ import annotations
import time
import logging
from typing import List, Dict, Any

from schemas.state import DoclamarState
from schemas.models import ParsedChunkSchema
from utils.llm_provider import get_llm

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000


def _build_numbered_context(chunks: List[ParsedChunkSchema]) -> tuple[str, List[Dict]]:
    parts = []
    citation_map = []
    total = 0

    for i, chunk in enumerate(chunks):
        ref_id = i + 1
        file_name = chunk.metadata.get("file_name", "unknown")
        file_path = chunk.metadata.get("file_path", "")
        chunk_idx = chunk.metadata.get("chunk_index", "0")

        block = f"[{ref_id}] Source: {file_name} (chunk {chunk_idx})\n{chunk.content}"

        if total + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 150:
                parts.append(block[:remaining])
            break

        parts.append(block)
        total += len(block)

        citation_map.append({
            "ref_id": ref_id,
            "file_name": file_name,
            "file_path": file_path,
            "chunk_index": chunk_idx,
            "score": chunk.score,
            "preview": chunk.content[:120].replace("\n", " "),
        })

    return "\n\n---\n\n".join(parts), citation_map


def summarization_node(state: DoclamarState) -> DoclamarState:
    t0 = time.time()
    logger.info("[SummarizationNode] Generating answer with citations...")

    raw = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    if not raw:
        return {
            **state,
            "final_answer": "I could not find relevant information in the provided documents.",
            "source_files": [],
            "citations": [],
        }

    chunks = [ParsedChunkSchema(**{k: v for k, v in c.items() if k != "score"}) for c in raw]
    for chunk, raw_chunk in zip(chunks, raw):
        chunk.score = raw_chunk.get("score")

    context, citation_map = _build_numbered_context(chunks)
    query = state["query"]

    system = """You are an expert document analysis assistant.
Answer questions using the provided document excerpts.
The excerpts are numbered [1], [2], etc. and come from the actual document.
You MUST search ALL excerpts carefully before concluding that information is absent.
Sections like conclusions, future work, and results are often in later numbered excerpts.
Cite sources inline using [N] reference numbers.
Only say information is missing if you have genuinely checked every excerpt provided."""

    prompt = f"""User Question: {query}

Document Excerpts:
{context}

Instructions:
- Use clear paragraphs and bullet points to make the summary highly readable.
- Answer precisely and cite every claim with its reference number inline e.g. "Machine learning is a subset of AI [1]."
- If multiple sources support a claim, cite all of them e.g. [1][3].
- Synthesize across sources where relevant.
- End with a "References" section, listing each [N] → filename on a brand new line.

Answer:"""

    try:
        llm = get_llm()
        answer = llm.generate(prompt, system=system)

        source_files = list({
            c.metadata.get("file_path", "")
            for c in chunks
            if c.metadata.get("file_path")
        })
        scores = [c.score for c in chunks if c.score is not None]

        elapsed = round(time.time() - t0, 3)
        timings = dict(state.get("node_timings") or {})
        timings["summarization"] = elapsed

        evaluation = {
            "chunks_used": len(chunks),
            "sources_cited": len(source_files),
            "avg_retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "top_score": round(max(scores), 4) if scores else 0.0,
        }

        logger.info(f"[SummarizationNode] Done in {elapsed}s | {len(citation_map)} citation(s)")

        return {
            **state,
            "final_answer": answer.strip(),
            "source_files": source_files,
            "citations": citation_map,
            "evaluation": evaluation,
            "error": None,
            "node_timings": timings,
        }

    except Exception as e:
        logger.error(f"[SummarizationNode] Error: {e}")
        return {
            **state,
            "final_answer": f"Generation failed: {e}",
            "source_files": [],
            "citations": [],
            "error": str(e),
        }
