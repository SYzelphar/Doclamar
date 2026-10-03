import os
import shutil
import time

import pytest

from doclamar.engine import Engine
from doclamar.store import Scope, fts_query
from tests.conftest import SAMPLE_PDF, FakeEmbedder, FakeLLM, FakeReranker


def _names(engine, folder):
    return sorted(os.path.basename(p) for p in engine.store.files_in(Scope(folder=str(folder))))


def test_sync_indexes_supported_files_and_skips_the_rest(engine, docs):
    job = engine.indexer.sync_folder(str(docs))
    assert job.state == "done"
    assert job.files_found == 3
    assert _names(engine, docs) == ["meeting.docx", "notes.txt", "recipe.md"]
    summary = engine.store.folder_summary(str(docs))
    assert summary["files_indexed"] == 3 and summary["chunks"] >= 3 and summary["synced_at"]


def test_resync_is_incremental(engine, docs):
    engine.indexer.sync_folder(str(docs))
    calls = engine.embedder.calls

    job = engine.indexer.sync_folder(str(docs))
    assert job.to_index == 0 and engine.embedder.calls == calls  # nothing re-embedded

    notes = docs / "notes.txt"
    notes.write_text("Completely new text about transformers.", encoding="utf-8")
    future = time.time() + 5
    os.utime(notes, (future, future))
    (docs / "sub" / "recipe.md").unlink()

    job = engine.indexer.sync_folder(str(docs))
    assert job.to_index == 1 and job.removed == 1
    assert _names(engine, docs) == ["meeting.docx", "notes.txt"]
    hits = engine.retriever.search("transformers", Scope(folder=str(docs)))
    assert hits and hits[0].name == "notes.txt" and "transformers" in hits[0].text


def test_problem_files_are_recorded_not_fatal(engine, tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "broken.pdf").write_bytes(b"not really a pdf")
    import pypdfium2 as pdfium

    blank = pdfium.PdfDocument.new()
    blank.new_page(612, 792)
    blank.save(str(folder / "scan.pdf"))
    blank.close()
    shutil.copy(SAMPLE_PDF, folder / "paper.pdf")

    job = engine.indexer.sync_folder(str(folder))
    assert job.state == "done" and job.failed == 1
    summary = engine.store.folder_summary(str(folder))
    statuses = {os.path.basename(i["path"]): i["status"] for i in summary["issues"]}
    assert statuses == {"broken.pdf": "error", "scan.pdf": "no_text"}
    assert summary["files_indexed"] == 1


def test_too_large_files_are_skipped(settings, docs):
    settings.max_file_mb = 0
    e = Engine(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    try:
        e.indexer.sync_folder(str(docs))
        summary = e.store.folder_summary(str(docs))
        assert summary["files_indexed"] == 0
        assert {i["status"] for i in summary["issues"]} == {"too_large"}
    finally:
        e.close()


def test_folder_scope_does_not_leak_into_sibling_prefix(engine, tmp_path):
    # The old code used startswith(), so "ML" also matched "ML_old".
    for name, text in [("ML", "alpha topic"), ("ML_old", "alpha topic secret"), ("a_b%c", "alpha wild")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "doc.txt").write_text(text, encoding="utf-8")
        engine.indexer.sync_folder(str(tmp_path / name))

    hits = engine.retriever.search("alpha", Scope(folder=str(tmp_path / "ML")))
    assert {os.path.dirname(h.path) for h in hits} == {str(tmp_path / "ML")}
    # Paths containing SQL LIKE wildcards are matched literally.
    hits = engine.retriever.search("alpha", Scope(folder=str(tmp_path / "a_b%c")))
    assert len(hits) == 1 and "a_b%c" in hits[0].path


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_folder_scope_is_case_insensitive_on_windows(engine, docs):
    engine.indexer.sync_folder(str(docs))
    assert engine.retriever.search("CorrMSE", Scope(folder=str(docs).upper()))


def test_file_scope(engine, docs):
    engine.indexer.sync_folder(str(docs))
    hits = engine.retriever.search("pancakes flour", Scope(files=[str(docs / "notes.txt")]))
    assert hits and all(h.name == "notes.txt" for h in hits)


@pytest.mark.parametrize("query", ['what is "quoted" text?', "AND OR NOT", "col:on (paren) * ^", "???", ""])
def test_keyword_search_tolerates_fts_syntax(engine, docs, query):
    engine.indexer.sync_folder(str(docs))
    engine.store.keyword_search(query, Scope(folder=str(docs)), 10)  # must not raise


def test_fts_query_drops_stopwords():
    assert fts_query("What is the loss function?") == '"loss" OR "function"'
    assert fts_query("what is the") is None


def test_changing_index_format_rebuilds(settings, docs):
    e = Engine(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    e.indexer.sync_folder(str(docs))
    e.close()
    assert e.store.folder_summary(str(docs))["files_indexed"] == 3

    settings.chunk_chars = 500  # e.g. a new chunking strategy or embedding model
    e2 = Engine(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    try:
        assert e2.store.folder_summary(str(docs))["files_indexed"] == 0
        assert e2.indexer.sync_folder(str(docs)).to_index == 3
    finally:
        e2.close()


def test_background_job_reports_progress(engine, docs):
    job = engine.indexer.start(str(docs))
    assert engine.indexer.start(str(docs)) is job or job.state == "done"  # no duplicate jobs
    deadline = time.time() + 20
    while job.active and time.time() < deadline:
        time.sleep(0.05)
    assert job.state == "done" and job.processed == job.to_index == 3
    with pytest.raises(FileNotFoundError):
        engine.indexer.start(str(docs / "missing"))
