"""PDF service — text/table/image extraction with per-page OCR fallback.

- Native text extraction via PyMuPDF (fast, layout-preserving).
- Table extraction via pdfplumber on demand.
- Image extraction (reaction schemes / structures) via PyMuPDF.
- Scanned-page detection: when native text is sparse, OCR the page image.
- Preserves original page numbers (PDF page index + 1).
- Validates page count / password-protection and raises friendly errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from backend.config import settings
from backend.services.ocr_service import ocr_available, ocr_image
from backend.utils import chunk_text

logger = logging.getLogger("pca.pdf")


class PDFError(Exception):
    pass


class PasswordProtectedPDFError(PDFError):
    pass


class TooManyPagesError(PDFError):
    pass


@dataclass
class ExtractedImage:
    index: int
    width: int
    height: int
    path: str | None = None
    caption: str = ""


@dataclass
class ExtractedPage:
    page_no: int  # 1-based original page number
    text: str = ""
    ocr_status: str = "none"  # native | ocr | mixed | empty
    ocr_text: str = ""
    images: list[ExtractedImage] = field(default_factory=list)
    image_path: str | None = None


@dataclass
class PDFExtraction:
    filename: str
    page_count: int
    pages: list[ExtractedPage]
    metadata: dict = field(default_factory=dict)
    ocr_used: bool = False


def extract_pdf(pdf_path: Path, job_id: int | None = None) -> PDFExtraction:
    """Extract everything from a patent PDF.

    Args:
        pdf_path: path to the uploaded PDF.
        job_id: used only to name stored page images (may be None for tests).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PDFError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        doc.close()
        raise PasswordProtectedPDFError("The PDF is password-protected. Please provide an unlocked PDF.")

    page_count = doc.page_count
    if page_count > settings.MAX_PAGES:
        doc.close()
        raise TooManyPagesError(
            f"PDF has {page_count} pages (limit is {settings.MAX_PAGES}). "
            "Please split the patent into parts or provide a smaller document."
        )

    pdf_path.name
    pages: list[ExtractedPage] = []
    ocr_used = False
    _page_dir: Path | None = None
    if job_id is not None:
        _page_dir = settings.PAGE_IMAGE_DIR / f"job_{job_id}"
        _page_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(page_count):
        page = doc.load_page(idx)
        epage = ExtractedPage(page_no=idx + 1)

        text = page.get_text("text")
        epage.text = text.strip()
        char_count = len(epage.text)

        # Sparse text (typical of scanned patents) -> OCR the rendered page.
        need_ocr = char_count < 120
        if need_ocr and ocr_available():
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
                png_bytes = pix.tobytes("png")
                pil_img = _png_bytes_to_image(png_bytes)
                ocr_text = ocr_image(pil_img)
                epage.ocr_text = ocr_text.strip()
                epage.ocr_status = "ocr" if not epage.text else "mixed"
                epage.text = (epage.text + "\n" + ocr_text).strip() if epage.text else ocr_text
                ocr_used = True
                if job_id is not None and _page_dir is not None:
                    _save_page_image(pix, _page_dir / f"page_{idx + 1}.png")
                    epage.image_path = str(_page_dir / f"page_{idx + 1}.png")
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for page %d: %s", idx + 1, exc)
                epage.ocr_status = "failed"
        elif need_ocr:
            epage.ocr_status = "empty" if not epage.text else "native"

        # Extract embedded images (structures / reaction schemes) regardless.
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image and base_image["width"] > 120 and base_image["height"] > 120:
                    epage.images.append(
                        ExtractedImage(
                            index=img_idx,
                            width=base_image["width"],
                            height=base_image["height"],
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("image extract page %d img %d failed: %s", idx + 1, img_idx, exc)

        pages.append(epage)

    metadata = {
        "title": doc.metadata.get("title") or "",
        "author": doc.metadata.get("author") or "",
        "subject": doc.metadata.get("subject") or "",
        "creator": doc.metadata.get("creator") or "",
    }
    doc.close()
    return PDFExtraction(
        filename=pdf_path.name,
        page_count=page_count,
        pages=pages,
        metadata=metadata,
        ocr_used=ocr_used,
    )


def extract_tables(pdf_path: Path, page_no: int) -> list[list[list[str]]]:
    """Extract tables from one page using pdfplumber."""
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            if page_no < 1 or page_no > len(pdf.pages):
                return []
            tbls = pdf.pages[page_no - 1].extract_tables() or []
            return tbls
    except Exception as exc:  # noqa: BLE001
        logger.warning("table extraction page %d failed: %s", page_no, exc)
        return []


def build_corpus(pages: list[ExtractedPage]) -> list[dict]:
    """Build the RAG corpus: chunked text with page + section metadata."""
    corpus: list[dict] = []
    for page in pages:
        if not page.text.strip():
            continue
        for i, chunk in enumerate(chunk_text(page.text)):
            corpus.append({
                "page_no": page.page_no,
                "chunk_index": i,
                "text": chunk,
                "section_kind": "other",
            })
    return corpus


def _png_bytes_to_image(png_bytes: bytes):
    from PIL import Image
    import io

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _save_page_image(pix, path: Path) -> None:
    pix.save(str(path))
