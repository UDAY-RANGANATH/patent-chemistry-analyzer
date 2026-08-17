"""SQLAlchemy ORM models for a patent analysis job.

Every chemical fact keeps page-level provenance so the UI can jump to the
patent page and the Word report can cite "Patent page N".

Tables:
  jobs          — one row per uploaded patent + analysis status/progress
  pages         — extracted text/image per page (original page numbers preserved)
  sections      — detected patent sections (claims / description / examples / process)
  compounds     — identified chemical entities + validated structure data
  reactions     — extracted reactions, conditions, yields, atom-mapping
  reaction_participants — many-to-many compounds ↔ reactions (with roles)
  stages        — manufacturing stage-by-stage reconstruction
  rag_chunks    — chunked patent text for retrieval (RAG) + page provenance
  provenance    — generic fact provenance (entity -> page -> evidence text)
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Table,
    Text,
    Column,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    original_filename: Mapped[str] = mapped_column(String(512))
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")  # uploaded|processing|complete|failed
    current_stage: Mapped[str] = mapped_column(String(128), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    patent_title: Mapped[str] = mapped_column(String(512), default="Not specified in patent")
    patent_number: Mapped[str] = mapped_column(String(128), default="Not specified in patent")
    assignee: Mapped[str] = mapped_column(String(512), default="Not specified in patent")
    inventors: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    filing_date: Mapped[str] = mapped_column(String(64), default="Not specified in patent")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    pages: Mapped[list["Page"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    sections: Mapped[list["Section"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    compounds: Mapped[list["Compound"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    reactions: Mapped[list["Reaction"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    stages: Mapped[list["Stage"]] = relationship(back_populates="job", cascade="all, delete-orphan", order_by="Stage.order_idx")


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("job_id", "page_no", name="uq_page"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    page_no: Mapped[int] = mapped_column(Integer)  # original PDF page number (1-based)
    text: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    ocr_status: Mapped[str] = mapped_column(String(16), default="none")  # none|native|ocr|mixed
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="pages")


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    kind: Mapped[str] = mapped_column(String(32))  # abstract|background|summary|claims|description|examples|process|manufacturing|other
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str] = mapped_column(String(512), default="")
    excerpt: Mapped[str] = mapped_column(Text, default="")

    job: Mapped["Job"] = relationship(back_populates="sections")


compound_reaction = Table(
    "reaction_participants",
    Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("reaction_id", Integer, ForeignKey("reactions.id"), nullable=False),
    Column("compound_id", Integer, ForeignKey("compounds.id"), nullable=False),
    Column("role", String(24), nullable=False),  # reactant|product|intermediate|reagent|catalyst|solvent
    Column("details", String(256), default=""),  # e.g. "1.2 eq" / "dropwise"
)


def reaction_roles(db: "Session", rxn_id: int) -> dict[int, str]:
    """Map compound_id -> role for a reaction (reads the association table)."""
    from sqlalchemy import select

    rows = db.execute(
        select(compound_reaction.c.compound_id, compound_reaction.c.role).where(
            compound_reaction.c.reaction_id == rxn_id
        )
    ).all()
    return {cid: role for cid, role in rows}


class Compound(Base):
    __tablename__ = "compounds"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    cid: Mapped[str] = mapped_column(String(32))  # C1, C2 ... per job
    name: Mapped[str] = mapped_column(String(512))          # as named in patent
    iupac_name: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    common_name: Mapped[str] = mapped_column(String(512), default="Not specified in patent")
    molecular_formula: Mapped[str] = mapped_column(String(128), default="Not specified in patent")
    molecular_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    smiles: Mapped[str | None] = mapped_column(Text, nullable=True)
    inchi: Mapped[str | None] = mapped_column(Text, nullable=True)
    inchikey: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cas_number: Mapped[str] = mapped_column(String(64), default="Not specified in patent")
    role: Mapped[str] = mapped_column(String(24), default="compound")  # reactant|intermediate|product|reagent|solvent|catalyst|...
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    source_basis: Mapped[str] = mapped_column(String(16), default="patent")  # patent|database|ai
    structure_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # json: {"db_conflicts": [...], "properties": {...}, "desc": {...}}
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    detected_from_image: Mapped[bool] = mapped_column(Boolean, default=False)

    job: Mapped["Job"] = relationship(back_populates="compounds")


class Reaction(Base):
    __tablename__ = "reactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    rid: Mapped[str] = mapped_column(String(32))  # R1, R2 ...
    type: Mapped[str] = mapped_column(String(64), default="Not specified in patent")
    name: Mapped[str] = mapped_column(String(256), default="")
    reactants_text: Mapped[str] = mapped_column(Text, default="")
    products_text: Mapped[str] = mapped_column(Text, default="")
    reagents: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    catalysts: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    solvents: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    temperature: Mapped[str] = mapped_column(String(128), default="Not specified in patent")
    pressure: Mapped[str] = mapped_column(String(128), default="Not specified in patent")
    time: Mapped[str] = mapped_column(String(128), default="Not specified in patent")
    atmosphere: Mapped[str] = mapped_column(String(128), default="Not specified in patent")
    yield_pct: Mapped[str] = mapped_column(String(64), default="Not specified in patent")
    workup: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reaction_smarts: Mapped[str | None] = mapped_column(Text, nullable=True)
    atom_map: Mapped[str | None] = mapped_column(Text, nullable=True)  # json: {"mapped_rxnsmiles":...}
    what_changed: Mapped[str | None] = mapped_column(Text, nullable=True)  # json summary
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="reactions")
    participants: Mapped[list["Compound"]] = relationship(
        secondary=compound_reaction, backref="in_reactions"
    )


class Stage(Base):
    __tablename__ = "stages"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    order_idx: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(512))
    purpose: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    starting_material: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    reagents: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    conditions: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    reaction: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    product: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    what_changed: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    why_required: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    chemistry: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    purification: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    yield_value: Mapped[str] = mapped_column(String(64), default="Not specified in patent")
    equipment: Mapped[str] = mapped_column(Text, default="Not specified in patent")
    patent_ref: Mapped[str] = mapped_column(String(128), default="Not specified in patent")
    scale: Mapped[str] = mapped_column(String(16), default="lab")  # lab|industrial|both|unknown

    job: Mapped["Job"] = relationship(back_populates="stages")


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    page_no: Mapped[int] = mapped_column(Integer)
    section_kind: Mapped[str] = mapped_column(String(32), default="other")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    keywords: Mapped[str] = mapped_column(Text, default="")  # space-joined lowercased tokens


class Provenance(Base):
    __tablename__ = "provenance"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    entity_type: Mapped[str] = mapped_column(String(32))  # compound|reaction|stage|fact
    entity_id: Mapped[int] = mapped_column(Integer)
    page_no: Mapped[int] = mapped_column(Integer)
    evidence: Mapped[str] = mapped_column(Text, default="")
