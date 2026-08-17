"""Unit tests for pipeline helpers (no network, no DB)."""

from backend.agents.pipeline import (
    _match_compound,
    _ns,
    _strip_stage_prefix,
    _tagged_chunks,
)


def test_strip_stage_prefix_variants():
    assert _strip_stage_prefix("Stage 1 - Esterification") == "Esterification"
    assert _strip_stage_prefix("Stage 2: Saponification") == "Saponification"
    assert _strip_stage_prefix("Stage 3 — Crystallization") == "Crystallization"
    assert _strip_stage_prefix("Esterification") == "Esterification"
    assert _strip_stage_prefix("stage 5 - x") == "x"


def test_ns():
    assert _ns(None) == "Not specified in patent"
    assert _ns("") == "Not specified in patent"
    assert _ns("   ") == "Not specified in patent"
    assert _ns("methanol") == "methanol"


def test_tagged_chunks_groups_pages():
    corpus = [
        {"page_no": 1, "text": "page one text", "section_kind": "abstract"},
        {"page_no": 2, "text": "page two text", "section_kind": "examples"},
        {"page_no": 3, "text": "page three text", "section_kind": "examples"},
        {"page_no": 4, "text": "page four text", "section_kind": "claims"},
    ]
    blocks = _tagged_chunks(corpus, pages_per_chunk=2)
    assert len(blocks) == 2
    assert "[Patent page 1]" in blocks[0] and "[Patent page 2]" in blocks[0]
    assert "[Patent page 3]" in blocks[1] and "[Patent page 4]" in blocks[1]


def test_tagged_chunks_drops_empty_pages():
    corpus = [{"page_no": 1, "text": "   ", "section_kind": "x"},
              {"page_no": 2, "text": "real", "section_kind": "x"}]
    blocks = _tagged_chunks(corpus, pages_per_chunk=2)
    assert len(blocks) == 1
    assert "real" in blocks[0]


def test_match_compound_exact():
    from types import SimpleNamespace

    mk = lambda n: SimpleNamespace(name=n, lower=n)
    compounds = {
        "methyl 4-hydroxybenzoate": mk("methyl 4-hydroxybenzoate"),
        "methanol": mk("methanol"),
    }
    assert _match_compound("Methyl 4-Hydroxybenzoate", compounds) is compounds["methyl 4-hydroxybenzoate"]
    assert _match_compound("methanol", compounds) is compounds["methanol"]


def test_match_compound_label_and_substring():
    from types import SimpleNamespace

    compounds = {"4-hydroxybenzoic acid": SimpleNamespace(name="4-hydroxybenzoic acid")}
    assert _match_compound("compound 3", compounds) is None  # label-style, resolve later
    assert _match_compound("4-hydroxybenzoic", compounds) is compounds["4-hydroxybenzoic acid"]
    assert _match_compound("zzz", compounds) is None
