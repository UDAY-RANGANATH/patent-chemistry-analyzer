"""Pydantic API schemas shared between backend and the JSON contracts.

Every field that a patent may not state defaults to "Not specified in patent".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: int
    filename: str
    status: str = "uploaded"


class PageOut(BaseModel):
    page_no: int
    text: str = ""
    ocr_status: str = "none"
    char_count: int = 0


class SectionOut(BaseModel):
    id: int
    kind: str
    page_start: int
    page_end: int
    heading: str = ""


class CompoundOut(BaseModel):
    id: int
    cid: str
    name: str
    iupac_name: str = "Not specified in patent"
    common_name: str = "Not specified in patent"
    molecular_formula: str = "Not specified in patent"
    molecular_weight: float | None = None
    smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    cas_number: str = "Not specified in patent"
    role: str = "compound"
    source_page: int | None = None
    source_text: str = ""
    confidence: float = 0.0
    source_basis: str = "patent"
    structure_valid: bool = False
    image_url: str | None = None
    extra: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class ReactionParticipant(BaseModel):
    cid: str
    name: str
    role: str
    details: str = ""


class ReactionOut(BaseModel):
    id: int
    rid: str
    type: str = "Not specified in patent"
    name: str = ""
    reagents: str = "Not specified in patent"
    catalysts: str = "Not specified in patent"
    solvents: str = "Not specified in patent"
    temperature: str = "Not specified in patent"
    pressure: str = "Not specified in patent"
    time: str = "Not specified in patent"
    atmosphere: str = "Not specified in patent"
    yield_pct: str = "Not specified in patent"
    workup: str = "Not specified in patent"
    source_page: int | None = None
    source_text: str = ""
    confidence: float = 0.0
    what_changed: dict[str, Any] | None = None
    participants: list[ReactionParticipant] = []

    model_config = {"from_attributes": True}


class StageOut(BaseModel):
    id: int
    order: int
    title: str
    purpose: str = "Not specified in patent"
    starting_material: str = "Not specified in patent"
    reagents: str = "Not specified in patent"
    conditions: str = "Not specified in patent"
    reaction: str = "Not specified in patent"
    product: str = "Not specified in patent"
    what_changed: str = "Not specified in patent"
    why_required: str = "Not specified in patent"
    chemistry: str = "Not specified in patent"
    purification: str = "Not specified in patent"
    yield_value: str = "Not specified in patent"
    equipment: str = "Not specified in patent"
    patent_ref: str = "Not specified in patent"
    scale: str = "lab"

    model_config = {"from_attributes": True}


class FlowNode(BaseModel):
    id: str
    kind: str = "compound"  # compound|stage|start|product
    compound: CompoundOut | None = None
    label: str = ""
    page: int | None = None
    stage: StageOut | None = None
    image_url: str | None = None
    x: float = 0
    y: float = 0


class FlowEdge(BaseModel):
    id: str
    source: str
    target: str
    reaction: ReactionOut | None = None
    label: str = ""


class FlowchartOut(BaseModel):
    job_id: int
    nodes: list[FlowNode]
    edges: list[FlowEdge]
    layout: str = "hierarchical"


class ManufacturingOut(BaseModel):
    job_id: int
    scale_summary: str = "Not specified in patent"
    raw_materials: list[str] = []
    process_units: list[str] = []
    equipment: list[str] = []
    stages: list[StageOut] = []
    notes: str = ""


class SourceOut(BaseModel):
    page_no: int
    text: str = ""
    image_url: str | None = None


class AnalysisSummary(BaseModel):
    job_id: int
    status: str
    progress: float
    current_stage: str = ""
    page_count: int
    filename: str
    patent_title: str = "Not specified in patent"
    patent_number: str = "Not specified in patent"
    assignee: str = "Not specified in patent"
    inventors: str = "Not specified in patent"
    filing_date: str = "Not specified in patent"
    sections: list[SectionOut] = []
    compound_count: int = 0
    reaction_count: int = 0
    stage_count: int = 0
    confidence: float = 0.0
    ocr_pages: int = 0
    database_conflicts: int = 0
    error: str | None = None


class ReportOut(BaseModel):
    filename: str
    url: str


class ProgressEvent(BaseModel):
    job_id: int
    stage: str
    progress: float
    detail: str = ""


class HealthOut(BaseModel):
    status: str = "ok"
    provider: str = "ollama"
    tesseract: bool = False
    rdkit: bool = False
    version: str = "1.0.0"
