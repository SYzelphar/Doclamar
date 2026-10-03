"""End-to-end retrieval with the real ONNX models (skipped until they are downloaded)."""
import shutil

import pytest

from doclamar.config import default_model_dir
from doclamar.engine import Engine
from doclamar.store import Scope
from tests.conftest import SAMPLE_PDF, FakeLLM

pytestmark = pytest.mark.models

if not any(default_model_dir().glob("models--*")):
    pytest.skip("models not downloaded (python scripts/download_models.py)", allow_module_level=True)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    from doclamar.config import Settings

    tmp = tmp_path_factory.mktemp("real")
    folder = tmp / "papers"
    folder.mkdir()
    shutil.copy(SAMPLE_PDF, folder / "paper.pdf")
    e = Engine(Settings(home=tmp / "home"), llm=FakeLLM(configured=False))
    e.indexer.sync_folder(str(folder))
    yield e, Scope(folder=str(folder))
    e.close()


@pytest.mark.parametrize("question, expected", [
    ("What loss function was used during training?", "CorrMSE"),
    ("Which dataset was used?", "GRID"),
    ("What optimizer was used?", "Adam"),
])
def test_finds_the_evidence(real, question, expected):
    engine, scope = real
    hits = engine.retriever.search(question, scope, top_k=3)
    assert any(expected in h.text for h in hits)


def test_unrelated_question_scores_below_not_found_threshold(real):
    engine, scope = real
    hits = engine.retriever.search("Who won the 2010 FIFA World Cup?", scope)
    assert hits[0].relevance < engine.settings.not_found_relevance
