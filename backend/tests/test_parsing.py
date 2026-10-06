import pytest

from doclamar.chunking import chunk_document
from doclamar.parsing import ParseError, ParsedDocument, Segment, is_heading_line, parse_file
from tests.conftest import SAMPLE_PDF


@pytest.mark.parametrize("line", [
    "ABSTRACT", "1. INTRODUCTION", "2.1. Data Preparation", "3.5 Ablation study",
    "Conclusion", "6. REFERENCES", "PROPOSED METHOD",
])
def test_heading_positives(line):
    assert is_heading_line(line)


@pytest.mark.parametrize("line", [
    # Ordinary body lines the old IGNORECASE regex treated as headings:
    "We use auditory spectrogram as spectral representation of",
    "improves the quality of reconstructed speech from the main",
    # Tables, footers, figure labels, sentences:
    "PESQ 2.76 2.81 2.92 2.88 2.33",
    "978-1-5386-4658-8/18/$31.00 ©2018 IEEE 2516 ICASSP 2018",
    "LSTM",
    "ICASSP 2018",
    "PESQ 1.54",
    "The network was trained for 100 epochs.",
    "3",
])
def test_heading_negatives(line):
    assert not is_heading_line(line)


def test_pdf_sections_and_pages():
    doc = parse_file(str(SAMPLE_PDF))
    headings = [s.text for s in doc.segments if s.is_heading]
    assert doc.num_pages == 5
    assert headings[:3] == ["LIP2AUDSPEC: SPEECH RECONSTRUCTION FROM SILENT LIP MOVEMENTS VIDEO",
                            "ABSTRACT", "1. INTRODUCTION"]
    assert "4. CONCLUSION" in headings and "3.5. Ablation study" in headings
    assert len(headings) == 16
    text = " ".join(s.text for s in doc.segments)
    assert "\x02" not in text
    assert "reconstructing intelligible speech" in text  # soft hyphen "recon-structing" re-joined
    assert all(s.page is not None for s in doc.segments)


def test_pdf_chunks_keep_title_pages_and_size():
    chunks = chunk_document(parse_file(str(SAMPLE_PDF)))
    assert chunks[0].text.startswith("LIP2AUDSPEC")  # preamble (title, authors) is kept
    assert "Hassan Akbari" in chunks[0].text
    assert all(len(c.text) <= 1000 for c in chunks)
    assert min(len(c.text) for c in chunks) >= 300
    assert all(c.page_start and c.page_end and c.page_start <= c.page_end for c in chunks)
    conclusion = [c for c in chunks if c.section == "4. CONCLUSION"]
    assert conclusion and "end-to-end" in conclusion[0].text.replace("‑", "-")


def test_chunks_split_on_sentences_with_overlap():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(80)]
    doc = ParsedDocument(path="x.txt", segments=[Segment("Section A", is_heading=True),
                                                 Segment(" ".join(sentences))])
    chunks = chunk_document(doc, chunk_chars=300, overlap_chars=80, min_chunk_chars=100)
    assert len(chunks) > 5
    for c in chunks:
        assert c.text.rstrip().endswith(".")
        assert c.section == "Section A"
    # consecutive chunks share their boundary sentence(s)
    first_last = chunks[0].text.split(". ")[-1]
    assert first_last.rstrip(".") in chunks[1].text


def test_overlap_does_not_cross_sections():
    body = " ".join(f"Alpha sentence {i} is here." for i in range(40))
    doc = ParsedDocument(path="x.txt", segments=[
        Segment("Part one", is_heading=True), Segment(body),
        Segment("Part two", is_heading=True), Segment("Beta is the only sentence in part two."),
    ])
    chunks = chunk_document(doc, chunk_chars=300, overlap_chars=100, min_chunk_chars=50)
    last = chunks[-1]
    assert last.section == "Part two"
    assert "Alpha" not in last.text


def test_docx_headings_and_tables(docs):
    doc = parse_file(str(docs / "sub" / "meeting.docx"))
    assert doc.segments[0].is_heading and doc.segments[0].text == "Meeting notes"
    assert any(s.text == "Owner | Shlok" for s in doc.segments)


def test_markdown_and_text(docs):
    md = parse_file(str(docs / "sub" / "recipe.md"))
    assert md.segments[0].is_heading and md.segments[0].text == "Pancakes"
    txt = parse_file(str(docs / "notes.txt"))
    assert txt.segments[0].is_heading and txt.segments[0].text == "TRAINING DETAILS"
    assert "CorrMSE" in txt.segments[1].text


def test_legacy_encoding(tmp_path):
    path = tmp_path / "old.txt"
    path.write_bytes("Café résumé naïve".encode("cp1252"))
    assert parse_file(str(path)).segments[0].text == "Café résumé naïve"


def test_bad_files(tmp_path):
    corrupt = tmp_path / "broken.pdf"
    corrupt.write_bytes(b"this is not a pdf")
    with pytest.raises(ParseError):
        parse_file(str(corrupt))
    with pytest.raises(ParseError):
        parse_file(str(tmp_path / "file.xyz"))
