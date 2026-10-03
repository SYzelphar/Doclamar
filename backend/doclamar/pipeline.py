"""The question-answering graph.

    condense ─► retrieve ─► (relevant?) ──yes──► generate
                   ▲            │ no, first try
                   └─ rewrite ◄─┘

* condense: only for follow-ups in a conversation; turns "what about the second
  one?" into a standalone search query (1 LLM call).
* rewrite: if nothing relevant came back, reword the query once and retry
  (1 LLM call). Most questions never take this path.
* generate: one LLM call that answers from numbered excerpts with [n] citations.

Typical cost: one LLM call per question (was N files + 4 in the old pipeline).
Nodes return only the keys they change; there is no accumulating reducer on the
retry counter (the old one doubled on every node).
"""
from __future__ import annotations

import logging
import operator
import re
import time
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import LLMClient, LLMError
from .retrieval import Hit, Retriever
from .store import Scope

logger = logging.getLogger(__name__)


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class RAGState(TypedDict, total=False):
    question: str
    history: List[dict]
    scope: Scope
    search_query: str
    hits: List[Hit]
    attempts: int
    answer: str
    citations: List[dict]
    status: str  # answered | not_found | no_llm | llm_error
    timings: Annotated[dict, _merge]


SYSTEM_PROMPT = """You are DocLAMAR, an assistant that answers questions using excerpts from the user's own documents.
Rules:
- Use only the numbered excerpts. If they do not contain the answer, say plainly that you could not find it in the documents. Do not guess or use outside knowledge.
- Cite every factual statement with its excerpt number in square brackets, e.g. [2] or [1][3].
- Lead with the direct answer (no "Answer:" label), then supporting detail. Use short paragraphs or bullet points and Markdown where it helps.
- If excerpts come from different files that disagree or describe different things, say which file each statement comes from.
- Do not add a references list at the end; the sources are shown to the user separately."""

CONDENSE_PROMPT = """Rewrite the user's latest message as a standalone search query for their documents, resolving pronouns and references using the conversation. Return only the query, nothing else.

Conversation:
{history}

Latest message: {question}

Standalone query:"""

REWRITE_PROMPT = """A search of the user's documents for the question below found nothing relevant. Rewrite it as a short keyword search query using different wording, synonyms or technical terms the documents might use. Return only the query, nothing else.

Question: {question}

Search query:"""


def _clean_query(text: str, fallback: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.strip().strip('"').strip("'").strip()
    return line[:300] or fallback


def _format_history(history: List[dict], limit: int = 600) -> str:
    lines = []
    for msg in history:
        who = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        lines.append(f"{who}: {content[:limit]}{'…' if len(content) > limit else ''}")
    return "\n".join(lines)


def format_excerpts(hits: List[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        where = " — ".join(x for x in (hit.name, hit.pages, hit.section) if x)
        blocks.append(f"[{n}] {where}\n{hit.text}")
    return "\n\n".join(blocks)


_ALT_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")  # 【1】 or 【1†L3-L5】


def normalize_citations(answer: str) -> str:
    """Some models (gpt-oss) cite as 【1】; the UI expects [1]."""
    return _ALT_CITATION.sub(lambda m: f"[{m.group(1)}]", answer)


def citations_for(hits: List[Hit], answer: str) -> List[dict]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [
        {
            "ref": n,
            "file": hit.name,
            "path": hit.path,
            "pages": hit.pages,
            "page": hit.page_start,
            "section": hit.section,
            "snippet": hit.text[:400],
            "relevance": hit.relevance,
            "cited": n in cited,
        }
        for n, hit in enumerate(hits, start=1)
    ]


def _top(hits: List[Hit]) -> float:
    if not hits:
        return -1.0
    return 1.0 if hits[0].relevance is None else hits[0].relevance


def build_pipeline(retriever: Retriever, llm: LLMClient, settings: Settings):
    def confident(hits: List[Hit]) -> bool:
        return _top(hits) >= settings.min_relevance

    def has_evidence(hits: List[Hit]) -> bool:
        return _top(hits) >= settings.not_found_relevance

    def condense(state: RAGState) -> dict:
        t0 = time.time()
        question, history = state["question"], state.get("history") or []
        query = question
        if history and llm.configured:
            try:
                query = _clean_query(
                    llm.generate(CONDENSE_PROMPT.format(history=_format_history(history), question=question),
                                 max_tokens=100, temperature=0),
                    question,
                )
            except LLMError as e:
                logger.warning("Condense failed, using the raw question: %s", e)
        return {"search_query": query, "attempts": 0, "timings": {"condense": round(time.time() - t0, 3)}}

    def retrieve(state: RAGState) -> dict:
        t0 = time.time()
        hits = retriever.search(state["search_query"], state["scope"])
        retry = bool(state.get("attempts"))
        if retry and _top(state.get("hits") or []) > _top(hits):
            hits = state["hits"]  # the rewrite made things worse; keep the first results
        key = "retrieve_retry" if retry else "retrieve"
        return {"hits": hits, "timings": {key: round(time.time() - t0, 3)}}

    def rewrite(state: RAGState) -> dict:
        t0 = time.time()
        query = state["search_query"]
        try:
            query = _clean_query(
                llm.generate(REWRITE_PROMPT.format(question=state["question"]), max_tokens=60, temperature=0),
                query,
            )
        except LLMError as e:
            logger.warning("Rewrite failed: %s", e)
        return {"search_query": query, "attempts": state.get("attempts", 0) + 1,
                "timings": {"rewrite": round(time.time() - t0, 3)}}

    def after_retrieve(state: RAGState) -> str:
        if confident(state.get("hits") or []):
            return "generate"
        if state.get("attempts", 0) < 1 and llm.configured:
            return "rewrite"
        return "generate"

    def generate(state: RAGState) -> dict:
        t0 = time.time()
        hits = state.get("hits") or []
        if not has_evidence(hits):
            return {
                "status": "not_found",
                "answer": "I couldn't find anything relevant to that in these documents. "
                          "Try rephrasing, or check that the right folder is selected and indexed.",
                "citations": [],
                "timings": {"generate": round(time.time() - t0, 3)},
            }
        if not llm.configured:
            return {
                "status": "no_llm",
                "answer": "Add an LLM API key in **Settings** to get a written answer. "
                          "Meanwhile, these are the most relevant passages I found:",
                "citations": citations_for(hits, ""),
                "timings": {"generate": round(time.time() - t0, 3)},
            }

        history = state.get("history") or []
        prompt = f"Excerpts:\n\n{format_excerpts(hits)}\n\n"
        if history:
            prompt += f"Conversation so far:\n{_format_history(history)}\n\n"
        prompt += f"Question: {state['question']}"
        try:
            answer = normalize_citations(llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1500))
            status = "answered"
        except LLMError as e:
            logger.error("Answer generation failed: %s", e)
            answer = f"⚠️ {e}\n\nHere are the most relevant passages I found:"
            status = "llm_error"
        return {
            "status": status,
            "answer": answer,
            "citations": citations_for(hits, answer if status == "answered" else ""),
            "timings": {"generate": round(time.time() - t0, 3)},
        }

    graph = StateGraph(RAGState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, {"rewrite": "rewrite", "generate": "generate"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
