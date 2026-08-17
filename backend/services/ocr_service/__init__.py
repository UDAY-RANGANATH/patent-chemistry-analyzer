"""OCR service — Tesseract via pytesseract.

Resolves the binary path from config -> PATH -> common install locations.
Degrades gracefully: `ocr_available()` reports capability so the pipeline can
warn instead of crashing when scanning a scanned patent without Tesseract.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from backend.config import settings

_CANDIDATE_PATHS = [
    r"C:\Users\Udayi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

_resolved: str | None = None


def tesseract_path() -> str | None:
    global _resolved
    if _resolved is not None:
        return _resolved
    if settings.TESSERACT_PATH and Path(settings.TESSERACT_PATH).exists():
        _resolved = settings.TESSERACT_PATH
        return _resolved
    found = shutil.which("tesseract")
    if found:
        _resolved = found
        return _resolved
    for p in _CANDIDATE_PATHS:
        if Path(p).exists():
            _resolved = p
            return _resolved
    _resolved = ""
    return None


def ocr_available() -> bool:
    return bool(tesseract_path())


def _prepare_image(img: Image.Image) -> Image.Image:
    """Normalize for better OCR: grayscale, autocontrast, optional upscale."""
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    w, h = img.size
    # Upscale small images — Tesseract works best around 300 DPI.
    if max(w, h) < 1600 and w > 0 and h > 0:
        scale = max(1.0, 1600 / max(w, h))
        if scale > 1.0 and scale < 3.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _deskew(img: Image.Image) -> Image.Image:
    """Simple rotation correction using cv2 if available (best-effort)."""
    try:
        import cv2

        arr = np.array(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            h, w = arr.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    except Exception:  # noqa: BLE001
        pass
    return img


def ocr_image(img: Image.Image, lang: str | None = None) -> str:
    """OCR a PIL image; returns extracted text ("" on failure)."""
    path = tesseract_path()
    if not path:
        return ""
    pytesseract.pytesseract.tesseract_cmd = path
    prepared = _deskew(_prepare_image(img))
    try:
        return pytesseract.image_to_string(prepared, lang=lang or settings.TESSERACT_LANG)
    except pytesseract.TesseractError as exc:
        # Fallback without language data weirdness
        try:
            return pytesseract.image_to_string(prepared)
        except Exception:  # noqa: BLE001
            raise exc
    except Exception:  # noqa: BLE001
        return ""
