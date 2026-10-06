"""Split a parsed document into overlapping, sentence-aligned chunks.

Chunks never cut a sentence in half, start fresh at a section boundary once
they have enough text, and remember which section and pages they came from so
answers can cite "p. 4" instead of "chunk 17".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .parsing import ParsedDocument


@dataclass
class Chunk:
    ordinal: int
    text: str
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    ocr: bool = False  # some of the text came from OCR


@dataclass
class _Unit:
    text: str
    page: Optional[int]
    section: Optional[str]
    is_heading: bool = False
    ocr: bool = False


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _split_long(text: str, limit: int) -> List[str]:
    """Hard-split text with no sentence breaks (tables, lists) at word boundaries."""
    pieces, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > limit:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
        while len(current) > limit:  # a single absurdly long token
            pieces.append(current[:limit])
            current = current[limit:]
    if current:
        pieces.append(current)
    return pieces


def _sentences(text: str, limit: int) -> List[str]:
    out = []
    for sentence in _SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        out.extend([sentence] if len(sentence) <= limit else _split_long(sentence, limit))
    return out


def _render(units: List[_Unit]) -> str:
    text = ""
    for u in units:
        if u.is_heading:
            text += f"\n{u.text}\n"
        else:
            text += ("" if not text or text.endswith("\n") else " ") + u.text
    return text.strip()


def chunk_document(
    doc: ParsedDocument,
    chunk_chars: int = 1000,
    overlap_chars: int = 150,
    min_chunk_chars: int = 300,
) -> List[Chunk]:
    unit_limit = max(100, chunk_chars - overlap_chars)
    units: List[_Unit] = []
    section: Optional[str] = None
    for seg in doc.segments:
        if seg.is_heading:
            section = seg.text[:200]
            units.append(_Unit(seg.text, seg.page, section, is_heading=True, ocr=seg.ocr))
        else:
            units.extend(_Unit(s, seg.page, section, ocr=seg.ocr) for s in _sentences(seg.text, unit_limit))

    chunks: List[Chunk] = []

    def emit(buf: List[_Unit]):
        if not any(not u.is_heading for u in buf):
            return  # headings alone carry no content
        pages = [u.page for u in buf if u.page is not None]
        # A heading inside the chunk means most of its content belongs to that section
        # (headings only get merged in when the text before them is short).
        headings = [u.section for u in buf if u.is_heading]
        chunks.append(Chunk(
            ordinal=len(chunks),
            text=_render(buf),
            section=headings[-1] if headings else next((u.section for u in buf if u.section), None),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            ocr=any(u.ocr for u in buf),
        ))

    buf: List[_Unit] = []
    size = 0
    carried = 0  # how many units at the start of buf are overlap from the previous chunk
    for unit in units:
        if unit.is_heading and size >= min_chunk_chars:
            # New section and we already have a decent chunk: close it cleanly.
            emit(buf)
            buf, size, carried = [], 0, 0
        elif unit.is_heading and buf and len(buf) == carried:
            # Only overlap from the previous section so far; don't drag it across the boundary.
            buf, size, carried = [], 0, 0
        elif buf and size + len(unit.text) + 1 > chunk_chars:
            emit(buf)
            tail: List[_Unit] = []
            tail_size = 0
            for prev in reversed(buf):
                if prev.is_heading or tail_size + len(prev.text) > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_size += len(prev.text) + 1
            buf, size, carried = tail, tail_size, len(tail)
        buf.append(unit)
        size += len(unit.text) + 1
    if buf:
        emit(buf)
    return chunks
