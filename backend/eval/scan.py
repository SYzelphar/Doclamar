"""Make an image-only ("scanned") copy of a PDF, for testing OCR."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional


def make_scanned_pdf(src: Path, dst: Path, pages: Optional[Iterable[int]] = None, dpi: int = 150,
                     jpeg_quality: int = 70) -> Path:
    """Render pages to greyscale JPEGs (like a cheap scanner) and save them as a PDF with no text layer."""
    import pypdfium2 as pdfium
    from PIL import Image

    pdf = pdfium.PdfDocument(str(src))
    try:
        images = []
        for i in (pages if pages is not None else range(len(pdf))):
            img = pdf[i].render(scale=dpi / 72).to_pil().convert("L")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=jpeg_quality)
            buf.seek(0)
            images.append(Image.open(buf).convert("RGB"))
    finally:
        pdf.close()
    images[0].save(dst, save_all=True, append_images=images[1:], resolution=dpi)
    return dst
