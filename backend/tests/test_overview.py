"""'What is this file about?' questions: no single passage answers them, so they get an overview."""
import pytest

from doclamar.engine import Engine
from doclamar.pipeline import is_overview_question
from doclamar.store import Scope, _spread
from tests.conftest import SAMPLE_PDF, FakeEmbedder, FakeLLM, FakeOcr, FakeReranker


@pytest.mark.parametrize("q", [
    "what is the file about", "What is this about?", "explain the document", "summarize this",
    "what are the main topics?", "what's this pdf about", "give me an overview", "tl;dr",
    "tell me about this document", "what does the paper cover", "key takeaways please",
])
def test_detects_whole_document_questions(q):
    assert is_overview_question(q)


@pytest.mark.parametrize("q", [
    "what is sustainable development", "what loss function was used", "who are the authors",
    "explain backpropagation", "what does table 2 show", "what is the file format of the data",
])
def test_specific_questions_use_normal_search(q):
    assert not is_overview_question(q)


def test_spread_covers_opening_and_sections_and_skips_references():
    sections = ["Title"] * 2 + ["1. Intro"] * 4 + ["2. Method"] * 6 + ["3. Results"] * 6 + ["4. Conclusion"] * 2 \
        + ["References"] * 5
    chunks = [{"id": i, "section": s} for i, s in enumerate(sections)]
    picked = _spread(chunks, 6)
    ids = [c["id"] for c in picked]
    assert ids == sorted(ids) and len(ids) == 6
    assert ids[0] == 0                                           # the opening
    assert {2, 6, 12, 18} <= set(ids)                            # each section's first chunk
    assert all(c["section"] != "References" for c in picked)
    assert [c["id"] for c in _spread(chunks[:3], 6)] == [0, 1, 2]  # short documents: everything


def _engine(settings, llm, reranker=None):
    return Engine(settings, embedder=FakeEmbedder(), reranker=reranker or FakeReranker(), llm=llm, ocr=FakeOcr())


@pytest.fixture
def paper_engine(settings, tmp_path):
    import shutil

    folder = tmp_path / "papers"
    folder.mkdir()
    shutil.copy(SAMPLE_PDF, folder / "paper.pdf")
    (folder / "notes.txt").write_text("Shopping list: eggs, flour, milk.", encoding="utf-8")
    llm = FakeLLM(answer="It is a paper about reconstructing speech from lip movements [1].")
    e = _engine(settings, llm)
    e.indexer.sync_folder(str(folder))
    yield e, llm, folder
    e.close()


def test_file_overview_question_gets_a_spread_of_the_document(paper_engine):
    e, llm, folder = paper_engine
    result = e.answer("what is the file about", Scope(files=[str(folder / "paper.pdf")]))
    assert result["status"] == "answered"
    assert len(llm.prompts) == 1 and "spread across the whole" in llm.prompts[0]  # no rewrite detour
    cites = result["citations"]
    # The overview spread, plus up to 4 passages that also matched the question itself.
    assert e.settings.overview_chunks <= len(cites) <= e.settings.overview_chunks + 4
    assert any(c["snippet"].startswith("LIP2AUDSPEC") for c in cites)  # includes the title/opening
    assert all("REFERENCES" not in (c["section"] or "") for c in cites)
    assert len({c["section"] for c in cites}) >= 6                   # covers many sections


def test_file_chat_never_says_not_found_from_scores_alone(settings, tmp_path):
    import shutil

    folder = tmp_path / "one"
    folder.mkdir()
    shutil.copy(SAMPLE_PDF, folder / "paper.pdf")
    llm = FakeLLM()
    e = _engine(settings, llm, reranker=FakeReranker(fixed=-12.0))  # every passage scores as irrelevant
    try:
        e.indexer.sync_folder(str(folder))
        file_result = e.answer("how was it received?", Scope(files=[str(folder / "paper.pdf")]))
        assert file_result["status"] == "answered" and file_result["citations"]
        assert "No passage matched the question closely" in llm.prompts[-1]

        folder_result = e.answer("how was it received?", Scope(folder=str(folder)))
        assert folder_result["status"] == "not_found"  # folder search still says so honestly
    finally:
        e.close()


def test_folder_overview_covers_each_file(paper_engine):
    e, llm, folder = paper_engine
    result = e.answer("what are these documents about?", Scope(folder=str(folder)))
    assert {c["file"] for c in result["citations"]} == {"paper.pdf", "notes.txt"}
