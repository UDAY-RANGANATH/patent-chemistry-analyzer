"""Patent service — section detection and patent bibliographic extraction.

Two layers:
  1. Heuristic layer (regex/keywords) — fast, deterministic, works offline.
  2. Optional LLM assist — refines section boundaries when available.

Sections detected: abstract, background, summary, detailed description,
claims, examples, manufacturing/process, other.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.services.pdf_service import ExtractedPage

logger = logging.getLogger("pca.patent")

_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("abstract", [r"\babstract\b", r"\bABSTRACT\b"]),
    ("background", [r"\bbackground of the (?:present )?invention\b", r"\bfield of the invention\b"]),
    ("summary", [r"\bsummary of the (?:present )?invention\b", r"\bdisclosure\b"]),
    ("description", [r"\bdetailed description of the (?:present )?invention\b", r"\bdetailed description\b"]),
    ("examples", [r"\bexamples?\b", r"\bpreparative examples?\b", r"\bexperimental section\b"]),
    ("claims", [r"\bwhat is claimed(?: is|:)?\b", r"\bclaims?\s*$", r"\bpatent claims\b"]),
    ("process", [r"\bmanufacturing (?:process|method|procedure)\b", r"\bprocess for preparing\b",
                 r"\bindustrial (?:process|preparation)\b", r"\bpreparation of compounds\b"]),
]

# Claims page markers commonly appear as "What is claimed is:" or on a title line.
_CLAIMS_LINE = re.compile(r"^\s*(what is claimed(?: is)?\s*[:.\]]?|claims\s*[:.\]]?)\s*$", re.IGNORECASE)
_EXAMPLE_LINE = re.compile(r"^\s*(example|preparation)\s+[0-9IVX]+", re.IGNORECASE)
# Strong manufacturing heading (own line) — preferred over an abstract mention of
# "process for preparing" that would otherwise pin the section to page 1.
_MANUFACTURING_LINE = re.compile(
    r"^\s*(?:industrial\s+)?(?:manufacturing|preparation|synthesis)\s+(?:process|method|procedure|route)[s]?\b",
    re.IGNORECASE,
)


@dataclass
class DetectedSection:
    kind: str
    page_start: int
    page_end: int
    heading: str = ""
    excerpt: str = ""


@dataclass
class PatentInfo:
    title: str = "Not specified in patent"
    patent_number: str = "Not specified in patent"
    assignee: str = "Not specified in patent"
    applicants: str = "Not specified in patent"
    inventors: str = "Not specified in patent"
    filing_date: str = "Not specified in patent"
    publication_date: str = "Not specified in patent"
    abstract: str = "Not specified in patent"


def detect_sections(pages: list[ExtractedPage]) -> list[DetectedSection]:
    """Heuristic section detection across pages."""
    hits: list[tuple[int, str]] = []  # (page_no, kind)
    for page in pages:
        text = page.text
        for kind, patterns in _SECTION_PATTERNS:
            if kind in {h[1] for h in hits}:
                continue
            if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                hits.append((page.page_no, kind))

    # Claims/claims detection via dedicated line matching
    claims_page = None
    examples_start = None
    manufacturing_page = None
    for page in pages:
        for line in page.text.splitlines():
            if claims_page is None and _CLAIMS_LINE.match(line.strip()):
                claims_page = page.page_no
            if examples_start is None and _EXAMPLE_LINE.match(line.strip()):
                examples_start = page.page_no
            if manufacturing_page is None and _MANUFACTURING_LINE.match(line.strip()):
                manufacturing_page = page.page_no

    kinds = [k for _, k in hits]
    if claims_page is not None and "claims" not in kinds:
        hits.append((claims_page, "claims"))
    if examples_start is not None and "examples" not in kinds:
        hits.append((examples_start, "examples"))

    # Prefer a real manufacturing heading over an abstract mention of the
    # "process for preparing" that would otherwise pin the section to page 1.
    if manufacturing_page is not None:
        hits = [(p, ("process" if k == "process" else k)) for p, k in hits]
        process_hits = [p for p, k in hits if k == "process"]
        if process_hits:
            hits = [(manufacturing_page, "process") if k == "process" else (p, k) for p, k in hits]
            hits.sort()
        else:
            hits.append((manufacturing_page, "process"))
            hits.sort()

    hits.sort()
    sections: list[DetectedSection] = []
    total_pages = pages[-1].page_no if pages else 0

    # Fill page ranges between section start markers.
    for i, (page_no, kind) in enumerate(hits):
        end = (hits[i + 1][0] - 1) if i + 1 < len(hits) else total_pages
        excerpt = _page_excerpt(pages, page_no)
        sections.append(DetectedSection(kind=kind, page_start=page_no,
                                        page_end=max(page_no, end), excerpt=excerpt))

    return sections


def extract_patent_info(pages: list[ExtractedPage]) -> PatentInfo:
    """Extract bibliographic data (first pages usually contain these)."""
    first_6000 = "\n".join(p.text for p in pages[:8])[:6000]
    info = PatentInfo()

    title = re.search(r"(?:TITLE|Title)\s*[:\-]?\s*(.+)", first_6000)
    if title:
        info.title = title.group(1).strip()[:500]

    patent_no = re.search(r"\b(?:US|EP|WO|CN|JP)\s?[\d,./]{4,}", first_6000)
    if patent_no:
        info.patent_number = patent_no.group(0).strip()

    assignee = re.search(r"Assignee\s*[:\-]?\s*(.+)", first_6000)
    if assignee:
        info.assignee = assignee.group(1).strip()[:500]

    inventor = re.search(r"Inventor[^:]*\s*[:\-]?\s*(.+)", first_6000)
    if inventor:
        info.inventors = inventor.group(1).strip()[:500]

    filing = re.search(r"Filed\s*[:\-]?\s*(.+)", first_6000)
    if filing:
        info.filing_date = filing.group(1).strip()[:120]

    return info


def _page_excerpt(pages: list[ExtractedPage], page_no: int, limit: int = 200) -> str:
    for p in pages:
        if p.page_no == page_no:
            return p.text[:limit]
    return ""
