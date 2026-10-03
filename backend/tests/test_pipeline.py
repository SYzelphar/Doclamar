from doclamar.engine import Engine
from doclamar.pipeline import normalize_citations
from doclamar.store import Scope
from tests.conftest import FakeEmbedder, FakeLLM, FakeReranker


def _engine(settings, llm=None, reranker=None):
    return Engine(settings, embedder=FakeEmbedder(), reranker=reranker or FakeReranker(), llm=llm or FakeLLM())


def test_answer_uses_one_llm_call_and_marks_cited_sources(engine, docs, fake_llm):
    engine.indexer.sync_folder(str(docs))
    result = engine.answer("Which loss function and learning rate were used?", Scope(folder=str(docs)))
    assert result["status"] == "answered"
    assert len(fake_llm.prompts) == 1  # no condense / rewrite needed
    assert "Excerpts:" in fake_llm.prompts[0] and "CorrMSE" in fake_llm.prompts[0]
    first = result["citations"][0]
    assert first["cited"] and first["file"] == "notes.txt" and first["ref"] == 1
    assert {"path", "pages", "section", "snippet", "relevance"} <= set(first)


def test_unrelated_question_short_circuits_without_llm(settings, docs):
    llm = FakeLLM()
    e = _engine(settings, llm=llm, reranker=FakeReranker(fixed=-12.0))  # everything irrelevant
    try:
        e.indexer.sync_folder(str(docs))
        result = e.answer("Who won the world cup?", Scope(folder=str(docs)))
        assert result["status"] == "not_found"
        assert result["citations"] == []
        # one rewrite attempt is allowed, but never a generation call on no evidence
        assert all(not p.startswith("Excerpts:") for p in llm.prompts)
        assert len(llm.prompts) == 1 and "retrieve_retry" in result["timings"]
    finally:
        e.close()


def test_low_relevance_triggers_a_single_rewrite(settings, docs):
    llm = FakeLLM()
    e = _engine(settings, llm=llm, reranker=FakeReranker(fixed=-6.0))  # weak but non-zero evidence
    try:
        e.indexer.sync_folder(str(docs))
        result = e.answer("optimizer?", Scope(folder=str(docs)))
        assert result["search_query"] == "rewritten keyword query"
        assert result["status"] == "answered"
        assert [p.split("\n")[0][:20] for p in llm.prompts] == ["A search of the user", "Excerpts:"]
    finally:
        e.close()


def test_follow_up_questions_are_condensed(engine, docs, fake_llm):
    engine.indexer.sync_folder(str(docs))
    history = [{"role": "user", "content": "What loss did they use?"},
               {"role": "assistant", "content": "CorrMSE [1]."}]
    result = engine.answer("and why that one?", Scope(folder=str(docs)), history)
    assert result["search_query"] == "why was the CorrMSE loss function chosen"
    assert "Conversation so far" in fake_llm.prompts[-1]


def test_without_llm_key_the_passages_are_still_returned(settings, docs):
    e = _engine(settings, llm=FakeLLM(configured=False))
    try:
        e.indexer.sync_folder(str(docs))
        result = e.answer("CorrMSE loss", Scope(folder=str(docs)))
        assert result["status"] == "no_llm"
        assert result["citations"] and "Settings" in result["answer"]
    finally:
        e.close()


def test_llm_failure_is_reported_with_sources(settings, docs):
    from doclamar.llm import LLMError

    class Failing(FakeLLM):
        def generate(self, prompt, **kw):
            raise LLMError("groq rate limit reached. Wait a minute and try again.")

    e = _engine(settings, llm=Failing())
    try:
        e.indexer.sync_folder(str(docs))
        result = e.answer("CorrMSE loss", Scope(folder=str(docs)))
        assert result["status"] == "llm_error" and "rate limit" in result["answer"]
        assert result["citations"]
    finally:
        e.close()


def test_normalize_citations():
    assert normalize_citations("a【1】 b【2†L3-L5】 c[3]") == "a[1] b[2] c[3]"
