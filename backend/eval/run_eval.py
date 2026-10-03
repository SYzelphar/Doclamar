"""Evaluate DocLAMAR on the LIP2AUDSPEC paper hidden among distractor documents.

Retrieval metrics (default, offline, no LLM calls):
  file_hit@k   the paper is among the top-k retrieved chunks' files
  mrr          1 / rank of the first chunk from the paper
  precision    share of the top-k chunks that come from the paper
  ctx_recall   share of the reference answer's key terms found in the top-k chunks
               (a cheap proxy for "did we retrieve the evidence")
  negatives    top relevance for unanswerable questions (should stay below not_found_relevance)

With --answers it also runs the full pipeline (needs an LLM API key) and reports
ROUGE-L against the reference answers and whether the answer cites its sources.

    python -m eval.run_eval                 # from backend/
    python -m eval.run_eval --answers --json results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from eval.distractors import write_distractors  # noqa: E402
from eval.questions import QUESTIONS, TARGET_FILE, UNANSWERABLE  # noqa: E402

_STOP = set("""a an the of to in and or for with by on at from as is are was were be been that this
which it its into than then their there these those using used use between both each""".split())


def _terms(text: str) -> set:
    out = set()
    for tok in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()):
        if len(tok) < 3 and not tok[0].isdigit() or tok in _STOP:
            continue
        out.add(re.sub(r"(ing|ed|es|s)$", "", tok) if not tok[0].isdigit() else tok)
    return out


def rouge_l(hypothesis: str, reference: str) -> float:
    hyp = re.findall(r"\w+", hypothesis.lower())
    ref = re.findall(r"\w+", reference.lower())
    if not hyp or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for h in hyp:
        cur = [0] * (len(ref) + 1)
        for j, r in enumerate(ref, 1):
            cur[j] = prev[j - 1] + 1 if h == r else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(hyp), lcs / len(ref)
    return round(2 * p * r / (p + r), 4)


def build_corpus(root: Path) -> Path:
    shutil.copy(BACKEND / "eval" / "data" / TARGET_FILE, root / TARGET_FILE)
    write_distractors(root)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--answers", action="store_true", help="also generate answers with the LLM")
    parser.add_argument("--json", help="write per-question results to this file")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds to wait between LLM questions (free-tier rate limits)")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="doclamar-eval-"))
    os.environ["DOCLAMAR_HOME"] = str(work / "home")
    corpus = work / "corpus"
    corpus.mkdir()
    build_corpus(corpus)

    from rich.console import Console
    from rich.table import Table

    from doclamar.engine import Engine
    from doclamar.store import Scope

    console = Console()
    engine = Engine()
    try:
        t0 = time.time()
        job = engine.indexer.sync_folder(str(corpus))
        index_s = time.time() - t0
        summary = engine.store.folder_summary(str(corpus))
        console.print(
            f"Indexed {summary['files_indexed']}/{summary['files_total']} files, "
            f"{summary['chunks']} chunks in {index_s:.1f}s "
            f"(issues: {[Path(i['path']).name + ': ' + i['status'] for i in summary['issues']]})"
        )
        scope = Scope(folder=str(corpus))

        rows = []
        for q in QUESTIONS:
            t0 = time.time()
            hits = engine.retriever.search(q["query"], scope, top_k=args.top_k)
            latency = time.time() - t0
            from_target = [h.name == TARGET_FILE for h in hits]
            first = next((i for i, ok in enumerate(from_target) if ok), None)
            ref_terms = _terms(q["reference"])
            found = _terms(" ".join(h.text for h in hits))
            row = {
                "query": q["query"],
                "file_hit": first is not None,
                "mrr": round(1 / (first + 1), 4) if first is not None else 0.0,
                "precision": round(sum(from_target) / max(len(hits), 1), 4),
                "ctx_recall": round(len(ref_terms & found) / max(len(ref_terms), 1), 4),
                "top_relevance": hits[0].relevance if hits else None,
                "retrieval_s": round(latency, 3),
            }
            if args.answers:
                time.sleep(args.pause)
                t0 = time.time()
                result = engine.answer(q["query"], scope)
                row.update(
                    answer=result["answer"],
                    rouge_l=rouge_l(result["answer"], q["reference"]),
                    cited=bool(re.search(r"\[\d+\]", result["answer"])),
                    answer_s=round(time.time() - t0, 2),
                )
            rows.append(row)

        negatives = []
        for query in UNANSWERABLE:
            hits = engine.retriever.search(query, scope, top_k=args.top_k)
            negatives.append({"query": query, "top_relevance": hits[0].relevance if hits else None})

        table = Table(title=f"Retrieval (top-{args.top_k})")
        for col in ("Question", "hit", "MRR", "prec", "ctx", "rel", "sec"):
            table.add_column(col)
        if args.answers:
            table.add_column("R-L")
            table.add_column("cited")
        for r in rows:
            cells = [r["query"][:48], "✓" if r["file_hit"] else "✗", str(r["mrr"]), str(r["precision"]),
                     str(r["ctx_recall"]), str(r["top_relevance"]), str(r["retrieval_s"])]
            if args.answers:
                cells += [str(r["rouge_l"]), "✓" if r["cited"] else "✗"]
            table.add_row(*cells)
        console.print(table)

        def mean(key):
            return round(statistics.mean(r[key] for r in rows), 4)

        totals = {
            "file_hit@k": mean("file_hit"),
            "mrr": mean("mrr"),
            "precision": mean("precision"),
            "ctx_recall": mean("ctx_recall"),
            "median_retrieval_s": round(statistics.median(r["retrieval_s"] for r in rows), 3),
            "index_s": round(index_s, 2),
            "negatives_max_relevance": max((n["top_relevance"] or 0) for n in negatives),
            "not_found_threshold": engine.settings.not_found_relevance,
            "rewrite_threshold": engine.settings.min_relevance,
        }
        if args.answers:
            totals["rouge_l"] = mean("rouge_l")
            totals["cited"] = mean("cited")
            totals["median_answer_s"] = round(statistics.median(r["answer_s"] for r in rows), 2)
        console.print(totals)

        if args.json:
            Path(args.json).write_text(json.dumps({"totals": totals, "questions": rows,
                                                   "negatives": negatives}, indent=2))
        return 0
    finally:
        engine.close()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
