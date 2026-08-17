"""Unit tests for shared utilities."""

from backend.utils import chunk_text, coerce_str


def test_coerce_str_scalars():
    assert coerce_str(None) == ""
    assert coerce_str("x") == "x"
    assert coerce_str(5) == "5"
    assert coerce_str(5.5) == "5.5"


def test_coerce_str_list_and_dict():
    assert coerce_str(["a", "b"]) == "a; b"
    assert coerce_str({"k": "v"}) == "k: v"
    assert coerce_str([]) == ""


def test_chunk_text_short_passthrough():
    assert chunk_text("short", size=1200) == ["short"]


def test_chunk_text_splits_and_overlaps():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 340 for c in chunks)
    assert "".join(chunks).replace(" ", "").startswith("word0word1")


def test_chunk_text_blank():
    assert chunk_text("   ") == []
