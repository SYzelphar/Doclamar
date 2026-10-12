"""'What files are in this folder?' is answered from the index, not from passages."""
import pytest

from doclamar.pipeline import is_inventory_question
from doclamar.store import Scope


@pytest.mark.parametrize("q", [
    "what all files and folders exist in this folder", "list the files", "which files are there",
    "how many files are in this folder", "what's in this folder", "show me the folder structure",
    "what documents do I have", "list subfolders", "how many pdfs do i have",
])
def test_detects_folder_questions(q):
    assert is_inventory_question(q)


@pytest.mark.parametrize("q", [
    "which file talks about normalization", "what files mention ER diagrams", "list the ACID properties",
    "explain the file system", "what is a folder in an OS", "how many documents mention deadlock",
])
def test_content_questions_are_not_folder_questions(q):
    assert not is_inventory_question(q)


def test_folder_listing(engine, docs, fake_llm):
    (docs / "model_weights.pkl").write_bytes(b"x")
    (docs / "sub" / "deep").mkdir()
    (docs / "sub" / "deep" / "empty.txt").write_text("   ", encoding="utf-8")  # no text
    engine.indexer.sync_folder(str(docs))
    (docs / "late_arrival.md").write_text("# Added after indexing", encoding="utf-8")

    result = engine.answer("what all files and folders exist in this folder", Scope(folder=str(docs)))
    answer = result["answer"]
    assert result["status"] == "answered" and result["citations"] == [] and fake_llm.prompts == []
    assert answer.startswith("**docs** contains 5 documents across 3 folders:")
    assert "- 📁 **sub/** (3 documents)" in answer and "  - 📁 **deep/** (1 document)" in answer
    assert "    - empty.txt — no readable text" in answer
    assert "- late\\_arrival.md — not indexed yet" in answer      # markdown-escaped, not yet synced
    assert "- notes.txt" in answer and "meeting.docx" in answer
    assert "Also 2 other files I can't search" in answer and ".pkl" in answer and ".xlsx" in answer
    assert "secret.txt" not in answer and "node_modules" not in answer  # hidden/skipped folders stay out


def test_file_chat_explains_its_scope(engine, docs):
    engine.indexer.sync_folder(str(docs))
    result = engine.answer("list the files", Scope(files=[str(docs / "notes.txt")]))
    assert "This chat is about one file: **notes.txt**" in result["answer"]
    assert "Back to folder search" in result["answer"]


def test_empty_folder(engine, tmp_path):
    (tmp_path / "empty").mkdir()
    result = engine.answer("what files are there", Scope(folder=str(tmp_path / "empty")))
    assert "has no files DocLamar can read" in result["answer"]
