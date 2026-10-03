from __future__ import annotations
import re
import logging
from collections import Counter
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
from embeddings.embedder import embed_texts

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    query: str
    answer: str
    reference: str
    source_files: List[str]
    latency_total_s: float = 0.0
    node_timings: Dict[str, float] = field(default_factory=dict)
    ephemeral_indexing_time_s: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    semantic_similarity: float = 0.0
    hallucination_rate: float = 0.0
    retrieval_accuracy_mrr10: float = 0.0
    token_consumption: int = 0
    replan_activated: bool = False
    replan_success: bool = False
    replan_activation_rate: float = 0.0
    replan_success_rate: float = 0.0
    avg_graph_depth: float = 0.0
    retry_count: int = 0
    chunks_used: int = 0
    avg_retrieval_score: float = 0.0


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _count_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _rouge_n(hypothesis: str, reference: str, n: int) -> float:
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if not hyp or not ref:
        return 0.0
    hyp_ng = _count_ngrams(hyp, n)
    ref_ng = _count_ngrams(ref, n)
    overlap = sum(min(hyp_ng[k], ref_ng[k]) for k in hyp_ng if k in ref_ng)
    ref_total = sum(ref_ng.values())
    hyp_total = sum(hyp_ng.values())
    if ref_total == 0 or hyp_total == 0:
        return 0.0
    precision = overlap / hyp_total
    recall    = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _rouge_l(hypothesis: str, reference: str) -> float:
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if not hyp or not ref:
        return 0.0
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if hyp[i-1] == ref[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    precision = lcs_len / m
    recall    = lcs_len / n
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _semantic_similarity(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    embs = embed_texts([hypothesis, reference])
    a = embs[0] / (np.linalg.norm(embs[0]) + 1e-10)
    b = embs[1] / (np.linalg.norm(embs[1]) + 1e-10)
    return round(float(np.dot(a, b)), 4)


def _hallucination_rate(answer: str, source_chunks: List[str]) -> float:
    if not answer or not source_chunks:
        return 1.0
    answer_emb  = embed_texts([answer])
    chunk_embs  = embed_texts(source_chunks)
    a_norm = answer_emb / (np.linalg.norm(answer_emb, axis=1, keepdims=True) + 1e-10)
    c_norm = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-10)
    max_sim = float(np.max(c_norm @ a_norm.T))
    return round(max(0.0, 1.0 - max_sim), 4)


def _mrr_at_k(retrieved_files: List[str], relevant_files: List[str], k: int = 10) -> float:
    if not relevant_files or not retrieved_files:
        return 0.0
    for rank, f in enumerate(retrieved_files[:k], start=1):
        if any(rf.lower() in f.lower() or f.lower() in rf.lower() for rf in relevant_files):
            return round(1.0 / rank, 4)
    return 0.0


def _count_tokens(text: str) -> int:
    return len(text.split())


def _graph_depth(node_timings: Dict[str, float]) -> float:
    node_order = ["routing", "parsing", "retrieval", "reranking", "summarization"]
    return float(sum(1 for n in node_order if n in node_timings))


def evaluate_result(
    result: Dict[str, Any],
    reference_answer: str = "",
    relevant_files: Optional[List[str]] = None,
) -> EvalResult:
    query        = result.get("query", "")
    answer       = result.get("final_answer") or ""
    sources      = result.get("source_files") or []
    timings      = result.get("node_timings") or {}
    evaluation   = result.get("evaluation") or {}
    reranked     = result.get("reranked_chunks") or []
    source_contents = [c.get("content", "") for c in reranked]
    retry_count  = result.get("retry_count", 0)

    latency_total  = sum(timings.values())
    indexing_time  = timings.get("retrieval", 0.0)

    r1      = _rouge_n(answer, reference_answer, 1) if reference_answer else 0.0
    r2      = _rouge_n(answer, reference_answer, 2) if reference_answer else 0.0
    rl      = _rouge_l(answer, reference_answer)    if reference_answer else 0.0
    sem_sim = _semantic_similarity(answer, reference_answer) if reference_answer else 0.0
    hall    = _hallucination_rate(answer, source_contents)   if source_contents else 1.0
    mrr     = _mrr_at_k(sources, relevant_files or [], k=10)

    total_tokens      = _count_tokens(query) + _count_tokens(answer)
    replan_activated  = retry_count > 0
    replan_success    = replan_activated and bool(answer and "not contain" not in answer.lower())

    return EvalResult(
        query=query, answer=answer, reference=reference_answer, source_files=sources,
        latency_total_s=round(latency_total, 3),
        node_timings=timings,
        ephemeral_indexing_time_s=round(indexing_time, 3),
        rouge_1=r1, rouge_2=r2, rouge_l=rl,
        semantic_similarity=sem_sim,
        hallucination_rate=hall,
        retrieval_accuracy_mrr10=mrr,
        token_consumption=total_tokens,
        replan_activated=replan_activated,
        replan_success=replan_success,
        replan_activation_rate=1.0 if replan_activated else 0.0,
        replan_success_rate=1.0 if replan_success else 0.0,
        avg_graph_depth=_graph_depth(timings),
        retry_count=retry_count,
        chunks_used=evaluation.get("chunks_used", 0),
        avg_retrieval_score=evaluation.get("avg_retrieval_score", 0.0),
    )


def run_eval_suite(
    test_cases: List[Dict],
    root_path: str,
    top_k: int = 5,
    specific_files: Optional[List[str]] = None,
) -> List[EvalResult]:
    """
    Run evaluation suite.

    Args:
        test_cases:      List of {query, reference_answer, relevant_files}
        root_path:       Root directory to search documents in
        top_k:           Number of chunks to retrieve
        specific_files:  Optional list of absolute file paths to inject directly
                         into the pipeline, bypassing routing. Use this when you
                         already know which files to evaluate against.
    """
    from graph.builder import build_graph
    graph = build_graph()
    results = []

    for case in test_cases:
        query           = case["query"]
        reference       = case.get("reference_answer", "")
        relevant_files  = case.get("relevant_files", [])

        logger.info(f"[Eval] Running: {query}")

        from schemas.models import DocumentSchema
        import os
        from pathlib import Path

        injected_docs = None
        if specific_files:
            injected_docs = []
            for fp in specific_files:
                if os.path.exists(fp):
                    injected_docs.append(DocumentSchema(
                        file_path=fp,
                        file_name=Path(fp).name,
                        file_type=Path(fp).suffix.lstrip(".").lower(),
                        size_kb=round(os.path.getsize(fp) / 1024, 2),
                    ).model_dump())

        initial_state = {
            "query":               query,
            "root_path":           root_path,
            "top_k":               top_k,
            "routing_plan":        None,
            "candidate_documents": injected_docs,
            "parsed_chunks":       None,
            "retrieved_chunks":    None,
            "reranked_chunks":     None,
            "final_answer":        None,
            "source_files":        None,
            "citations":           None,
            "search_stats":        None,
            "evaluation":          None,
            "error":               None,
            "retry_count":         0,
            "node_timings":        {},
        }

        result = graph.invoke(initial_state)
        result["query"] = query

        er = evaluate_result(result, reference, relevant_files)
        results.append(er)

        logger.info(
            f"[Eval] R1={er.rouge_1} R2={er.rouge_2} RL={er.rouge_l} "
            f"Sem={er.semantic_similarity} Hall={er.hallucination_rate} "
            f"MRR={er.retrieval_accuracy_mrr10} Tok={er.token_consumption} "
            f"Lat={er.latency_total_s}s Depth={er.avg_graph_depth} Replan={er.replan_activated}"
        )

    return results


def aggregate_metrics(results: List[EvalResult]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {}

    def avg(vals):
        return round(sum(vals) / n, 4)

    replan_activated = [r for r in results if r.replan_activated]

    return {
        "end_to_end_latency_s":      avg([r.latency_total_s for r in results]),
        "ephemeral_indexing_time_s": avg([r.ephemeral_indexing_time_s for r in results]),
        "token_consumption_avg":     avg([r.token_consumption for r in results]),
        "hallucination_rate":        avg([r.hallucination_rate for r in results]),
        "rouge_1":                   avg([r.rouge_1 for r in results]),
        "rouge_2":                   avg([r.rouge_2 for r in results]),
        "rouge_l":                   avg([r.rouge_l for r in results]),
        "semantic_similarity":       avg([r.semantic_similarity for r in results]),
        "retrieval_accuracy_mrr10":  avg([r.retrieval_accuracy_mrr10 for r in results]),
        "replan_activation_rate":    round(len(replan_activated) / n, 4),
        "replan_success_rate":       round(sum(1 for r in replan_activated if r.replan_success) / max(len(replan_activated), 1), 4),
        "avg_graph_depth":           avg([r.avg_graph_depth for r in results]),
    }


def print_eval_report(results: List[EvalResult]):
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    from rich import box

    console = Console()
    agg = aggregate_metrics(results)

    per_q = Table(title="Per-Query Results", box=box.MARKDOWN)
    per_q.add_column("Query",  style="cyan", max_width=25)
    per_q.add_column("R-1",   style="green",   width=6)
    per_q.add_column("R-2",   style="green",   width=6)
    per_q.add_column("R-L",   style="green",   width=6)
    per_q.add_column("Sem",   style="green",   width=6)
    per_q.add_column("Hall",  style="red",     width=6)
    per_q.add_column("MRR",   style="yellow",  width=6)
    per_q.add_column("Tok",   style="magenta", width=5)
    per_q.add_column("Lat(s)", style="magenta", width=7)
    per_q.add_column("Idx(s)", style="dim",    width=7)
    per_q.add_column("Depth", style="dim",     width=6)
    per_q.add_column("Replan", style="red",    width=7)

    for r in results:
        per_q.add_row(
            r.query[:25],
            str(r.rouge_1), str(r.rouge_2), str(r.rouge_l),
            str(r.semantic_similarity), str(r.hallucination_rate),
            str(r.retrieval_accuracy_mrr10), str(r.token_consumption),
            str(r.latency_total_s), str(r.ephemeral_indexing_time_s),
            str(r.avg_graph_depth), "Y" if r.replan_activated else "N",
        )

    console.print(per_q)
    console.print()
    console.print(Rule("[bold]Aggregated Metrics[/bold]"))

    agg_t = Table(box=box.SIMPLE)
    agg_t.add_column("Metric", style="cyan")
    agg_t.add_column("Value",  style="green")

    labels = {
        "end_to_end_latency_s":      "End-to-End Query Latency (s)",
        "ephemeral_indexing_time_s": "Ephemeral Indexing Time (s)",
        "token_consumption_avg":     "Token Consumption (avg)",
        "hallucination_rate":        "Hallucination Rate",
        "rouge_1":                   "ROUGE-1",
        "rouge_2":                   "ROUGE-2",
        "rouge_l":                   "ROUGE-L",
        "semantic_similarity":       "Semantic Similarity",
        "retrieval_accuracy_mrr10":  "Retrieval Accuracy MRR@10",
        "replan_activation_rate":    "Replan Activation Rate",
        "replan_success_rate":       "Replan Success Rate",
        "avg_graph_depth":           "Average Graph Depth",
    }

    for key, label in labels.items():
        agg_t.add_row(label, str(agg.get(key, 0.0)))

    console.print(agg_t)
