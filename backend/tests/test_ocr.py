"""OCR: scanned PDFs and images become searchable text."""
import os
import re

import pytest

from doclamar.engine import Engine
from doclamar.ocr import OcrEngine
from doclamar.parsing import parse_file
from doclamar.store import Scope
from tests.conftest import FakeEmbedder, FakeLLM, FakeOcr, FakeReranker, make_scanned_pdf


def _engine(settings, ocr):
    return Engine(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM(), ocr=ocr)


def _status(engine, path):
    row = engine.store.get_file(str(path))
    return row["status"], row["ocr_pages"], row["error"]


# ------------------------------------------------------------ fake engine (fast)

def test_scanned_pdf_waits_for_ocr_then_is_picked_up_without_changing(settings, tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    scan = make_scanned_pdf(folder / "invoice.pdf")

    settings.ocr_enabled = False  # DOCLAMAR_OCR=0
    off = _engine(settings, ocr=None)
    off.indexer.sync_folder(str(folder))
    status, ocr_pages, error = _status(off, scan)
    assert status == "needs_ocr" and ocr_pages == 0 and "OCR is turned off" in error
    off.close()

    settings.ocr_enabled = True
    ocr = FakeOcr()
    on = _engine(settings, ocr=ocr)
    job = on.indexer.sync_folder(str(folder))  # file untouched, still re-read once OCR works
    assert job.to_index == 1 and ocr.calls == 1
    assert _status(on, scan)[:2] == ("indexed", 1)
    hits = on.retriever.search("invoice 4471 consulting", Scope(folder=str(folder)))
    assert hits and hits[0].ocr and "4471" in hits[0].text

    assert on.indexer.sync_folder(str(folder)).to_index == 0  # and not again after that
    on.close()


def test_ocr_failure_on_a_page_is_recorded_not_fatal(settings, tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    scan = make_scanned_pdf(folder / "bad.pdf")
    e = _engine(settings, ocr=FakeOcr(fail=True))
    try:
        assert e.indexer.sync_folder(str(folder)).state == "done"
        status, ocr_pages, error = _status(e, scan)
        assert status == "no_text" and ocr_pages == 1 and "even with OCR" in error
    finally:
        e.close()


def test_blank_pdf_pages_are_not_sent_to_ocr(settings, tmp_path):
    import pypdfium2 as pdfium

    folder = tmp_path / "blank"
    folder.mkdir()
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(612, 792)
    pdf.save(str(folder / "blank.pdf"))
    pdf.close()
    ocr = FakeOcr()
    e = _engine(settings, ocr=ocr)
    try:
        e.indexer.sync_folder(str(folder))
        assert ocr.calls == 0
        assert _status(e, folder / "blank.pdf")[0] == "no_text"
    finally:
        e.close()


def test_ocr_flag_reaches_citations(engine, tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    make_scanned_pdf(folder / "invoice.pdf")
    engine.indexer.sync_folder(str(folder))
    result = engine.answer("invoice number 4471", Scope(folder=str(folder)))
    assert result["citations"][0]["ocr"] is True
    assert "scanned text (OCR)" in engine.llm.prompts[-1]
    assert engine.store.folder_summary(str(folder))["files_ocr"] == 1


# ------------------------------------------------------------ real RapidOCR

@pytest.fixture(scope="module")
def real_ocr():
    ocr = OcrEngine()
    if not ocr.available:
        pytest.skip(f"RapidOCR unavailable: {ocr.unavailable_reason}")
    return ocr


def test_reads_an_image_file(real_ocr, tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1400, 360), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=44)
    draw.text((40, 60), "Quarterly revenue grew by 12 percent.", fill="black", font=font)
    draw.text((40, 180), "The board approved the new budget.", fill="black", font=font)
    path = tmp_path / "whiteboard.jpg"
    # Stored sideways with an EXIF "rotate 90 deg clockwise to display" tag, like a phone photo.
    img.rotate(90, expand=True).save(path, exif=_exif_rotated())

    doc = parse_file(str(path), ocr=real_ocr)
    text = " ".join(s.text for s in doc.segments).lower()
    assert doc.ocr_pages == 1 and all(s.ocr for s in doc.segments)
    assert "quarterly revenue grew by 12 percent" in text
    assert text.index("revenue") < text.index("board")


def _exif_rotated():
    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    return exif


def test_scanned_two_column_page_keeps_reading_order(real_ocr, tmp_path):
    scan = make_scanned_pdf(tmp_path / "page1.pdf", pages=(0,))
    doc = parse_file(str(scan), ocr=real_ocr)
    assert doc.ocr_pages == 1
    headings = [s.text for s in doc.segments if s.is_heading]
    assert "ABSTRACT" in headings and "1. INTRODUCTION" in headings
    text = " ".join(s.text for s in doc.segments)
    # Left column (abstract) must come before the right column it sits next to.
    assert text.index("In this study, we propose") < text.index("followed by an LSTM")
    assert "reconstructing intelligible speech" in text  # OCR line-break hyphen repaired


@pytest.mark.skipif(os.environ.get("DOCLAMAR_SKIP_SLOW") == "1", reason="slow")
def test_scanned_paper_is_searchable_end_to_end(real_ocr, settings, tmp_path):
    folder = tmp_path / "library"
    folder.mkdir()
    make_scanned_pdf(folder / "scanned_paper.pdf", pages=(2, 3))  # Implementation + Ablation pages
    e = _engine(settings, ocr=real_ocr)
    try:
        e.indexer.sync_folder(str(folder))
        summary = e.store.folder_summary(str(folder))
        assert summary["files_indexed"] == 1 and summary["files_ocr"] == 1
        hits = e.retriever.search("Adam optimizer initial learning rate", Scope(folder=str(folder)))
        assert any(re.search(r"0\.0001", h.text) and h.ocr and h.page_start in (1, 2) for h in hits)
    finally:
        e.close()
