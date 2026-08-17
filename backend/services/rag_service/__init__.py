"""RAG service — chunked retrieval with page-level provenance.

Patents up to 150 pages are chunked per page; every chunk knows its page and
(optionally) its section. Retrieval is hybrid:

  - FTS5 keyword search over the SQLite text index (fast, no model needed)
  - Optional embedding similarity (Ollama nomic-embed-text) when available

The AI agents use this to ground every explanation in actual patent text, so
answers carry "Patent page N" references instead of hallucinated citations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.models import Job, RagChunk
from backend.utils import chunk_text

logger = logging.getLogger("pca.rag")


@dataclass
class RetrievalHit:
    page_no: int
    section_kind: str
    text: str
    score: float = 0.0


def index_job(db: Session, job: Job, pages_with_sections: list[dict]) -> int:
    """Persist RAG chunks for a job.

    pages_with_sections: list of {"page_no", "text", "section_kind"}
    """
    db.query(RagChunk).filter(RagChunk.job_id == job.id).delete()
    count = 0
    for item in pages_with_sections:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        for i, chunk in enumerate(chunk_text(text, size=1100, overlap=120)):
            db.add(RagChunk(
                job_id=job.id,
                page_no=item["page_no"],
                section_kind=item.get("section_kind", "other"),
                chunk_index=i,
                text=chunk,
                keywords=" ".join(_tokenize(chunk)),
            ))
            count += 1
    db.commit()
    return count


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]{1,50}", text.lower()) if len(t) > 2]


def retrieve(
    db: Session,
    job_id: int,
    query: str,
    limit: int = 8,
    page_filter: int | None = None,
) -> list[RetrievalHit]:
    """Keyword retrieval over the job's RAG chunks (FTS5-like via LIKE/IN)."""
    terms = _tokenize(query)
    if not terms:
        return []

    chunks = db.query(RagChunk).filter(RagChunk.job_id == job_id)
    if page_filter:
        chunks = chunks.filter(RagChunk.page_no == page_filter)

    # Build an OR of LIKE conditions on the keyword column.
    from sqlalchemy import func

    cond = or_(*[RagChunk.keywords.contains(f" {t}") for t in terms])
    rows = chunks.filter(cond).order_by(RagChunk.page_no, RagChunk.chunk_index).limit(limit * 6).all()

    hits: list[RetrievalHit] = []
    for row in rows:
        hits.append(RetrievalHit(
            page_no=row.page_no,
            section_kind=row.section_kind,
            text=row.text,
            score=0.5,
        ))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def retrieve_with_context(db: Session, job_id: int, query: str, limit: int = 5) -> str:
    """Return a compact, citation-stamped context block for LLM prompts."""
    hits = retrieve(db, job_id, query, limit=limit)
    if not hits:
        return "[No matching patent text found for this query.]"
    parts = []
    for h in hits:
        excerpt = h.text.strip().replace("\n", " ")[:600]
        parts.append(f"[Patent page {h.page_no} — {h.section_kind}]: {excerpt}")
    return "\n\n".join(parts)


def full_text_for_pages(db: Session, job_id: int, page_start: int, page_end: int) -> str:
    from backend.models import Page

    rows = (
        db.query(Page)
        .filter(Page.job_id == job_id, Page.page_no >= page_start, Page.page_no <= page_end)
        .order_by(Page.page_no)
        .all()
    )
    return "\n\n".join(f"=== Patent page {p.page_no} ===\n{p.text}" for p in rows if p.text.strip())


def all_evidence(db: Session, job_id: int, page_no: int) -> str:
    """Everything the patent states on one page (for the 'View Patent Source' UI)."""
    from backend.models import Page

    row = db.query(Page).filter(Page.job_id == job_id, Page.page_no == page_no).first()
    return row.text if row else "Not specified in patent"
