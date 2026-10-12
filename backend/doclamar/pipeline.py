"""The question-answering graph.

    condense ─► retrieve ─► (relevant?) ──yes──► generate
                   ▲            │ no, first try
                   └─ rewrite ◄─┘

* condense: only for follow-ups in a conversation; turns "what about the second
  one?" into a standalone search query (1 LLM call).
* retrieve: relevance search, or for "what is this about?"-style questions an
  overview: passages spread across the document(s) plus any relevant hits. No
  single passage answers such questions, so relevance search alone finds nothing.
* rewrite: if nothing relevant came back, reword the query once and retry
  (1 LLM call). Most questions never take this path.
* generate: one LLM call that answers from numbered excerpts with [n] citations.
  When chatting with one file it never answers "not found" from scores alone:
  the user picked the document, so it falls back to an overview of it.

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
    mode: str  # search | overview | fallback
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
- Excerpts marked "scanned text (OCR)" were read from images and may contain recognition errors; read past obvious typos, but don't guess at numbers you can't make out.
- Do not add a references list at the end; the sources are shown to the user separately."""

CONDENSE_PROMPT = """Rewrite the user's latest message as a standalone search query for their documents, resolving pronouns and references using the conversation. Return only the query, nothing else.

Conversation:
{history}

Latest message: {question}

Standalone query:"""

REWRITE_PROMPT = """A search of the user's documents for the question below found nothing relevant. Rewrite it as a short keyword search query using different wording, synonyms or technical terms the documents might use. Return only the query, nothing else.

Question: {question}

Search query:"""


_DOC = r"(?:file|document|doc|pdf|paper|article|book|report|chapter|notes?|text|slides?|thesis)s?"
_OVERVIEW = re.compile(
    r"\b(?:summari[sz]e|summary|summarise|overview|tl;?dr|gist|synopsis)\b"
    r"|\bwhat(?:'s| is| are)\s+(?:this|that|it|the|these|those)\s+(?:" + _DOC + r"\s+)?(?:all\s+)?about\b"
    r"|\b(?:explain|describe|walk me through)\s+(?:this|the|these|that|my)\s+" + _DOC + r"\b"
    r"|\btell me about\s+(?:this|the|these|that)\s+" + _DOC + r"\b"
    r"|\bwhat\s+(?:does|do)\s+(?:this|the|these|that)\s+" + _DOC + r"\s+(?:say|cover|contain|discuss|explain|talk about|deal with)\b"
    r"|\bwhat(?:'s| is)\s+in\s+(?:this|the|these|that)\s+" + _DOC + r"\b"
    r"|\b(?:main|key|important|major)\s+(?:points?|ideas?|topics?|takeaways?|themes?|concepts?|findings?|sections?)\b"
    r"|\b(?:topics|subjects)\s+(?:are\s+)?(?:covered|discussed)\b",
    re.IGNORECASE,
)


def is_overview_question(question: str) -> bool:
    """Questions about a whole document rather than a fact inside it."""
    return bool(_OVERVIEW.search(question))


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
        where = " — ".join(x for x in (hit.name, hit.pages, hit.section, "scanned text (OCR)" if hit.ocr else "") if x)
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
            "ocr": hit.ocr,
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
        scope = state["scope"]
        if is_overview_question(state["question"]):
            # Relevant passages first (for "summarise the results"), then a spread of the document(s).
            matched = [h for h in retriever.search(state["search_query"], scope, top_k=4)
                       if h.relevance is None or h.relevance >= settings.min_relevance]
            seen = {h.chunk_id for h in matched}
            hits = matched + [h for h in retriever.overview(scope) if h.chunk_id not in seen]
            return {"hits": hits, "mode": "overview", "timings": {"retrieve": round(time.time() - t0, 3)}}

        hits = retriever.search(state["search_query"], scope)
        retry = bool(state.get("attempts"))
        if retry and _top(state.get("hits") or []) > _top(hits):
            hits = state["hits"]  # the rewrite made things worse; keep the first results
        key = "retrieve_retry" if retry else "retrieve"
        return {"hits": hits, "mode": "search", "timings": {key: round(time.time() - t0, 3)}}

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
        mode = state.get("mode", "search")
        scope = state["scope"]
        if not has_evidence(hits) and scope.files:
            # Chatting with one document: never answer "not found" from scores alone.
            hits, mode = retriever.overview(scope), "fallback"
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
        prompt = ""
        if mode == "overview":
            prompt += ("These excerpts are spread across the whole of the document(s) to give an overview"
                       f"{_folder_note(retriever, scope, hits)}. Use them to describe what the document(s)"
                       " cover, organised by topic.\n\n")
        elif mode == "fallback":
            prompt += ("No passage matched the question closely, so these excerpts are spread across the"
                       " document. Answer from them if they allow it; otherwise say the document doesn't"
                       " seem to cover it and mention what it does cover.\n\n")
        prompt += f"Excerpts:\n\n{format_excerpts(hits)}\n\n"
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

    def _folder_note(retriever: Retriever, scope: Scope, hits: List[Hit]) -> str:
        if scope.files:
            return ""
        total = len(retriever.store.scope_file_ids(scope))
        shown = len({h.path for h in hits})
        return f" (they come from {shown} of the {total} files in the folder)" if total > shown else ""

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
