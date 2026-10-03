"""Turn files into an ordered list of text segments (headings + body) with page numbers.

One parser is shared by folder search and single-document chat, so both see
exactly the same text.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


class ParseError(Exception):
    """The file could not be read (corrupt, encrypted, unsupported)."""


@dataclass
class Segment:
    text: str
    page: Optional[int] = None  # 1-based; None for formats without pages
    is_heading: bool = False


@dataclass
class ParsedDocument:
    path: str
    segments: List[Segment] = field(default_factory=list)
    num_pages: Optional[int] = None

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.segments)


# ---------------------------------------------------------------------------
# Heading detection for plain-text lines (PDF / TXT have no style information)
# ---------------------------------------------------------------------------

_NUMBER_PREFIX = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVX]{1,5}\.)\s+")
_KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "related work", "prior work",
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experiments", "experiment", "experimental setup", "experimental results",
    "evaluation", "results", "results and discussion", "discussion", "analysis",
    "conclusion", "conclusions", "future work", "limitations", "summary",
    "references", "bibliography", "acknowledgements", "acknowledgments", "appendix",
}


def is_heading_line(line: str) -> bool:
    s = line.strip()
    if not 3 <= len(s) <= 90:
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3 or len(s.split()) > 12:
        return False
    if s[-1] in ".,;":
        return False
    # Number-heavy lines are table rows, footers or equations, not headings.
    if len(letters) / len([c for c in s if not c.isspace()]) < 0.5:
        return False
    core = _NUMBER_PREFIX.sub("", s)
    if core.lower().rstrip(":") in _KNOWN_SECTIONS:
        return True
    # ALL-CAPS lines ("PROPOSED METHOD"). Case-sensitive on purpose: the old regex
    # used IGNORECASE and matched almost every line of body text. A lone caps word
    # is usually a figure label or acronym ("LSTM"), so require two words.
    if (len(letters) >= 4 and len(core.split()) >= 2
            and sum(c.isupper() for c in letters) / len(letters) > 0.9):
        return True
    # Numbered title-case headings ("3.2 Network Architecture").
    if _NUMBER_PREFIX.match(s) and core[:1].isupper() and len(core.split()) <= 8:
        return True
    return False


# ---------------------------------------------------------------------------
# Text clean-up helpers
# ---------------------------------------------------------------------------

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f￾￿]")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # pdfium marks a soft (line-break) hyphen as \x02: "recon\x02\nstructing".
    text = re.sub(r"\x02\s*\n\s*", "", text)
    text = _CONTROL.sub("", text)
    return text


def _join_lines(lines: List[str]) -> str:
    """Join wrapped lines of one paragraph back into prose."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[:1].isalpha():
            out += line  # "state-of-the-" + "art"
        else:
            out += " " + line
    return out


def _segments_from_lines(lines: List[str], page: Optional[int]) -> List[Segment]:
    segments: List[Segment] = []
    body: List[str] = []

    def flush():
        text = _join_lines(body)
        if text:
            segments.append(Segment(text=text, page=page))
        body.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if is_heading_line(stripped):
            flush()
            segments.append(Segment(text=stripped, page=page, is_heading=True))
        else:
            body.append(stripped)
    flush()
    return segments


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------

def _repeated_margin_lines(pages: List[List[str]]) -> set:
    """Running headers/footers: lines that appear at a page edge on most pages."""
    if len(pages) < 3:
        return set()
    counts: Counter = Counter()
    for lines in pages:
        edge = [l.strip() for l in lines[:2] + lines[-2:] if l.strip()]
        counts.update(set(edge))
    threshold = max(3, len(pages) // 2)
    return {line for line, n in counts.items() if n >= threshold}


def _read_pdf(path: str) -> ParsedDocument:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as e:
        raise ParseError(f"cannot open PDF ({e})") from e

    try:
        page_lines: List[List[str]] = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                page_lines.append(_clean(textpage.get_text_bounded()).split("\n"))
            finally:
                textpage.close()
                page.close()
    finally:
        pdf.close()

    margins = _repeated_margin_lines(page_lines)
    doc = ParsedDocument(path=path, num_pages=len(page_lines))
    for number, lines in enumerate(page_lines, start=1):
        kept = [
            l for l in lines
            if l.strip() and l.strip() not in margins and not l.strip().isdigit()
        ]
        doc.segments.extend(_segments_from_lines(kept, page=number))
    return doc


def _read_docx(path: str) -> ParsedDocument:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(path)
    except Exception as e:  # python-docx raises several unrelated types
        raise ParseError(f"cannot open DOCX ({e})") from e

    doc = ParsedDocument(path=path)
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = _clean(para.text).strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            heading = style.startswith("Heading") or style == "Title"
            doc.segments.append(Segment(text=text, is_heading=heading))
        elif tag == "tbl":
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    value = _clean(cell.text).strip().replace("\n", " ")
                    if value and (not cells or cells[-1] != value):  # merged cells repeat
                        cells.append(value)
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                doc.segments.append(Segment(text="\n".join(rows)))
    return doc


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")


def _read_text(path: str, markdown: bool) -> ParsedDocument:
    with open(path, "rb") as fh:
        text = _clean(_decode(fh.read()))

    doc = ParsedDocument(path=path)
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        if markdown:
            body: List[str] = []
            for line in lines:
                m = _MD_HEADING.match(line)
                if m:
                    if body:
                        doc.segments.append(Segment(text=_join_lines(body)))
                        body = []
                    doc.segments.append(Segment(text=m.group(1), is_heading=True))
                else:
                    body.append(line)
            if body:
                doc.segments.append(Segment(text=_join_lines(body)))
        elif len(lines) == 1 and is_heading_line(lines[0]):
            doc.segments.append(Segment(text=lines[0].strip(), is_heading=True))
        else:
            doc.segments.append(Segment(text=_join_lines(lines)))
    return doc


def parse_file(path: str) -> ParsedDocument:
    ext = Path(path).suffix.lstrip(".").lower()
    if ext == "pdf":
        return _read_pdf(path)
    if ext == "docx":
        return _read_docx(path)
    if ext in ("txt", "md"):
        return _read_text(path, markdown=(ext == "md"))
    raise ParseError(f"unsupported file type: .{ext}")
