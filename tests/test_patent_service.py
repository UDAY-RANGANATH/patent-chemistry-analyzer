"""Unit tests for patent section detection and bibliographic extraction."""

from backend.services.patent_service import detect_sections, extract_patent_info
from backend.services.pdf_service import ExtractedPage


def _page(no: int, text: str) -> ExtractedPage:
    return ExtractedPage(page_no=no, text=text, ocr_status="text")


def test_detects_classic_sections():
    pages = [
        _page(1, "Abstract\nThe invention concerns a process for preparing esters."),
        _page(2, "Background of the invention\nEsters are widely used."),
        _page(3, "Summary of the invention\nWe provide a process."),
        _page(4, "Detailed description\nIn a preferred embodiment, methanol is used."),
        _page(5, "Example 1\nPreparation of methyl 4-hydroxybenzoate."),
        _page(6, "Example 2\nPreparation of sodium 4-hydroxybenzoate."),
        _page(7, "What is claimed is:\n1. A process as claimed."),
    ]
    kinds = [(s.kind, s.page_start) for s in detect_sections(pages)]
    assert ("abstract", 1) in kinds
    assert ("background", 2) in kinds
    assert ("summary", 3) in kinds
    assert ("description", 4) in kinds
    assert ("examples", 5) in kinds
    assert ("claims", 7) in kinds


def test_industrial_manufacturing_heading_wins_over_abstract_mention():
    """An 'INDUSTRIAL MANUFACTURING PROCESS' heading must override the abstract's
    'process for preparing' so the process section points at the real text."""
    pages = [
        _page(1, "Abstract\nImproved process for preparing paraben esters."),
        _page(2, "Example 1\nEsterification performed."),
        _page(3, "INDUSTRIAL MANUFACTURING PROCESS\nAt industrial scale, a 1000 L reactor is used."),
        _page(4, "What is claimed is:\n1. The process."),
    ]
    sections = detect_sections(pages)
    process = [s for s in sections if s.kind == "process"]
    assert len(process) == 1
    assert process[0].page_start == 3
    assert "1000 L reactor" in process[0].excerpt


def test_manufacturing_section_when_only_title_mention():
    pages = [
        _page(1, "Title: A manufacturing process for hydroxybenzoates"),
        _page(2, "Detailed description\nEsterification in methanol."),
    ]
    sections = detect_sections(pages)
    process = [s for s in sections if s.kind == "process"]
    assert process and process[0].page_start == 1


def test_section_ranges_never_regress():
    pages = [_page(i, f"page {i}") for i in range(1, 6)]
    pages[4] = _page(5, "What is claimed is:\n1. The process.\nINDUSTRIAL MANUFACTURING PROCESS\nReactor.")
    for s in detect_sections(pages):
        assert s.page_end >= s.page_start


def test_example_line_detection():
    pages = [_page(1, "Example 2\nSaponification step.")]
    kinds = [s.kind for s in detect_sections(pages)]
    assert "examples" in kinds


def test_extract_patent_info():
    pages = [
        _page(1, "TITLE: PROCESS FOR THE PREPARATION OF METHYL 4-HYDROXYBENZOATE\n"
                 "Patent No: US 12,345,678 B2\nAssignee: Paracelsus Pharmaceuticals\n"
                 "Inventor: Alan T. Berzelius\nFiled: March 4, 2023"),
    ]
    info = extract_patent_info(pages)
    assert "METHYL 4-HYDROXYBENZOATE" in info.title
    assert "12,345,678" in info.patent_number
    assert info.assignee == "Paracelsus Pharmaceuticals"
    assert "Berzelius" in info.inventors
    assert "March 4, 2023" in info.filing_date


def test_no_sections_returns_empty():
    assert detect_sections([]) == []
