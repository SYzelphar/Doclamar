"""OCR for scanned PDF pages and images.

Engine: RapidOCR (PP-OCR models on ONNX Runtime — the same runtime the search
models use; the OCR models ship inside the pip package, so it works offline).

RapidOCR returns text boxes roughly top-to-bottom, which interleaves the columns
of a two-column scan. `layout_lines` rebuilds reading order: it finds column
gutters (recursively, so 3-column layouts work too), treats boxes that cross a
gutter (titles, full-width figures) as band separators, and merges boxes that
sit on the same baseline into one line.
"""
from __future__ import annotations

import logging
import os
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class OcrBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float = 1.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


# --------------------------------------------------------------------------
# Reading order
# --------------------------------------------------------------------------

def _find_gutter(boxes: Sequence[OcrBox], min_gap: float) -> Optional[tuple]:
    """Widest vertical strip in the middle of `boxes` that (almost) no box crosses."""
    left = min(b.x0 for b in boxes)
    right = max(b.x1 for b in boxes)
    span = right - left
    if span <= 0:
        return None
    lo, hi = left + 0.25 * span, left + 0.75 * span

    # Ignore boxes wider than half the region (titles, captions): they are what
    # we want to split *around*, not evidence against a gutter.
    narrow = [b for b in boxes if b.width < 0.5 * span]
    if not narrow:
        return None
    coverage = np.zeros(int(hi - lo) + 1, dtype=np.int32)
    for b in narrow:
        a, z = max(b.x0, lo) - lo, min(b.x1, hi) - lo
        if z > a:
            coverage[int(a):int(np.ceil(z)) + 1] += 1
    # Tolerate a stray box or two in the gutter (a centred page number, a figure label).
    free = coverage <= max(1, len(narrow) // 30)

    best, start = None, None
    for x, ok in enumerate(np.append(free, False)):
        if ok and start is None:
            start = x
        elif not ok and start is not None:
            if x - start >= min_gap and (best is None or x - start > best[1] - best[0]):
                best = (start, x)
            start = None
    if best is None:
        return None

    g0, g1 = lo + best[0], lo + best[1]
    n_left = sum(1 for b in narrow if b.x1 <= g0)
    n_right = sum(1 for b in narrow if b.x0 >= g1)
    # Both sides must hold a real column of text, not a stray page number.
    if min(n_left, n_right) < max(3, 0.15 * len(narrow)):
        return None
    return g0, g1


def _blocks(boxes: List[OcrBox], min_gap: float, depth: int = 0) -> List[List[OcrBox]]:
    """Split boxes into reading-order blocks (columns, separated by full-width bands)."""
    if depth > 3 or len(boxes) < 6:
        return [boxes]
    gutter = _find_gutter(boxes, min_gap)
    if gutter is None:
        return [boxes]
    centre = (gutter[0] + gutter[1]) / 2

    # Only boxes that straddle the gutter's centre separate bands; a column line that
    # pokes slightly into the gutter still belongs to its column.
    def straddles(b: OcrBox) -> bool:
        return b.x0 < centre - min_gap / 2 and b.x1 > centre + min_gap / 2

    crossing = sorted((b for b in boxes if straddles(b)), key=lambda b: b.cy)
    sides = [b for b in boxes if not straddles(b)]

    # Bands are bounded by the separators' centres so every box lands in exactly one.
    blocks: List[List[OcrBox]] = []
    top = float("-inf")
    for sep in crossing + [None]:
        bottom = sep.cy if sep is not None else float("inf")
        band = [b for b in sides if top <= b.cy < bottom]
        left = [b for b in band if (b.x0 + b.x1) / 2 < centre]
        right = [b for b in band if (b.x0 + b.x1) / 2 >= centre]
        for column in (left, right):
            if column:
                blocks.extend(_blocks(column, min_gap, depth + 1))
        if sep is not None:
            blocks.append([sep])
            top = sep.cy
    return blocks


def _lines(block: List[OcrBox], line_height: float) -> List[List[OcrBox]]:
    lines: List[List[OcrBox]] = []
    for box in sorted(block, key=lambda b: (b.cy, b.x0)):
        if lines:
            line = lines[-1]
            line_cy = statistics.fmean(b.cy for b in line)
            overlaps = any(box.x0 < b.x1 and b.x0 < box.x1 for b in line)
            if abs(box.cy - line_cy) <= 0.5 * line_height and not overlaps:
                line.append(box)
                continue
        lines.append([box])
    return [sorted(line, key=lambda b: b.x0) for line in lines]


def layout_lines(boxes: Iterable[OcrBox]) -> List[str]:
    """Text lines in reading order. Blank strings mark block (column/band) breaks."""
    boxes = [b for b in boxes if b.text.strip()]
    if not boxes:
        return []
    line_height = statistics.median(b.height for b in boxes)
    out: List[str] = []
    for block in _blocks(boxes, min_gap=max(4.0, 0.6 * line_height)):
        for line in _lines(block, line_height):
            out.append(" ".join(b.text.strip() for b in line))
        out.append("")
    return out[:-1]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class OcrUnavailable(RuntimeError):
    pass


class OcrEngine:
    """Thread-safe wrapper: one RapidOCR instance per worker thread."""

    def __init__(self, workers: Optional[int] = None):
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) // 4))
        self._local = threading.local()
        self._checked = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    def _create(self):
        from rapidocr import RapidOCR

        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        # Split the cores between the per-thread engines; letting each one grab
        # every core oversubscribes the CPU and made parallel OCR slower.
        threads = max(1, (os.cpu_count() or 2) // self.workers)
        return RapidOCR(params={
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })

    def _engine(self):
        engine = getattr(self._local, "engine", None)
        if engine is None:
            engine = self._local.engine = self._create()
        return engine

    @property
    def available(self) -> bool:
        if not self._checked:
            with self._lock:
                if not self._checked:
                    try:
                        self._engine()
                    except Exception as e:  # missing package, broken install, no CPU support…
                        self._error = f"OCR engine failed to load: {e}"
                        logger.warning(self._error)
                    self._checked = True
        return self._error is None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return None if self.available else self._error

    def recognize(self, image) -> List[OcrBox]:
        """`image`: a PIL image (RGB) or a BGR numpy array (pdfium's default render)."""
        if not self.available:
            raise OcrUnavailable(self._error)
        result = self._engine()(image)
        if result.boxes is None or not result.txts:
            return []
        boxes = []
        for quad, text, score in zip(result.boxes, result.txts, result.scores):
            pts = np.asarray(quad, dtype=float)
            boxes.append(OcrBox(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(),
                                text, float(score)))
        return boxes

    def read_lines(self, image) -> List[str]:
        return layout_lines(self.recognize(image))

    def map_ordered(self, fn: Callable[[T], R], items: Iterable[T]) -> Iterator[R]:
        """Run `fn` on worker threads, yielding results in input order with bounded memory."""
        if self.workers == 1:
            yield from map(fn, items)
            return
        with ThreadPoolExecutor(self.workers, thread_name_prefix="ocr") as pool:
            pending = []
            for item in items:
                pending.append(pool.submit(fn, item))
                if len(pending) >= self.workers * 2:
                    yield pending.pop(0).result()
            for future in pending:
                yield future.result()
