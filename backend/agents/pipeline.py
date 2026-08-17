"""PatentAnalysisPipeline — orchestrates the full analysis.

Stages (each persist progress to the job row via the callback):
  1. PDF extraction + OCR        (pdf_service + ocr_service)
  2. Patent section detection    (patent_service)
  3. RAG indexing                (rag_service)
  4. Chemical entity extraction  (ChemicalEntityAgent, chunked + parallel)
  5. Compound validation         (chemistry_service + structure_service + RDKit render)
  6. Reaction extraction         (ReactionAnalysisAgent, chunked)
  7. Reaction analysis           (reaction_service 'What Changed?')
  8. Manufacturing reconstruction(ManufacturingAgent)
  9. QC audit                    (validation_service)

The pipeline is designed for up to 150-page patents: text is chunked per page,
LLM calls are bounded per chunk, results are persisted incrementally so an API
failure never loses completed stages.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from backend.agents.entity_agent import ChemicalEntityAgent
from backend.agents.manufacturing_agent import ManufacturingAgent
from backend.agents.prompts import SUMMARY_SYSTEM, SUMMARY_USER
from backend.agents.reaction_agent import ReactionAnalysisAgent
from backend.models import (
    Compound,
    Job,
    Page,
    Provenance,
    RagChunk,
    Reaction,
    Section,
    Stage,
)
from backend.services.ai_service import AIService, system_message, user_message
from backend.services.chemistry_service import chemistry_service
from backend.services.flowchart_service import build_graph
from backend.services.patent_service import detect_sections, extract_patent_info
from backend.services.pdf_service import extract_pdf
from backend.services.rag_service import index_job
from backend.services.reaction_service import serialize, what_changed_text
from backend.services.structure_service import analyze, render_structure
from backend.services.validation_service import run_qc

logger = logging.getLogger("pca.pipeline")

ProgressCallback = Callable[[str, float, str], None]

_SKIP_ROLES = {"reagent", "catalyst", "solvent", "by-product", "additive", "base", "acid"}
_MAX_WORKERS = 2


class PipelineAborted(Exception):
    pass


def _notify(cb: ProgressCallback | None, stage: str, pct: float, detail: str = "") -> None:
    logger.info("pipeline [%s] %.0f%% %s", stage, pct * 100, detail)
    if cb:
        cb(stage, pct, detail)


def run_pipeline(
    db: Session,
    job: Job,
    pdf_path: Path,
    progress: ProgressCallback | None = None,
    ai: AIService | None = None,
) -> None:
    """Run the full analysis for a job. Updates job status/progress as it goes.

    `ai` is injectable for testing/offline runs; default is the configured provider.
    """
    ai = ai or AIService()

    # ------------------------------------------------------------------ #
    # 1. PDF + OCR
    # ------------------------------------------------------------------ #
    _notify(progress, "PDF uploaded", 0.02, job.filename)
    job.status = "processing"
    db.commit()

    _notify(progress, "PDF uploaded", 0.04, "pages detected")
    extraction = extract_pdf(pdf_path, job_id=job.id)
    job.page_count = extraction.page_count
    job.file_size_bytes = pdf_path.stat().st_size

    _notify(progress, "Text extracted", 0.10, f"{extraction.page_count} pages, "
            f"OCR={'yes' if extraction.ocr_used else 'no'}")

    for ep in extraction.pages:
        db.add(Page(
            job_id=job.id,
            page_no=ep.page_no,
            text=ep.text,
            ocr_status=ep.ocr_status,
            image_path=ep.image_path,
            ocr_text=ep.ocr_text,
        ))
    db.commit()
    _notify(progress, "Text extracted", 0.15, "pages persisted")

    # ------------------------------------------------------------------ #
    # 2. Sections
    # ------------------------------------------------------------------ #
    _notify(progress, "Analyzing patent structure", 0.18, "detecting sections")
    sections = detect_sections(extraction.pages)
    for s in sections:
        db.add(Section(job_id=job.id, kind=s.kind, page_start=s.page_start,
                       page_end=s.page_end, heading=s.heading, excerpt=s.excerpt))
    db.commit()
    _notify(progress, "Patent sections detected", 0.22,
            f"{len(sections)} sections found")

    # ------------------------------------------------------------------ #
    # 3. RAG index
    # ------------------------------------------------------------------ #
    _notify(progress, "Building search index", 0.24, "chunking patent text")
    sec_by_page: dict[int, str] = {}
    for s in sections:
        for p in range(s.page_start, s.page_end + 1):
            sec_by_page[p] = s.kind
    corpus = [
        {"page_no": ep.page_no, "text": ep.text,
         "section_kind": sec_by_page.get(ep.page_no, "other")}
        for ep in extraction.pages
    ]
    n_chunks = index_job(db, job, corpus)
    _notify(progress, "Search index ready", 0.28, f"{n_chunks} chunks indexed")

    # ------------------------------------------------------------------ #
    # 4. Entity extraction (chunked, parallel)
    # ------------------------------------------------------------------ #
    _notify(progress, "Extracting chemical entities", 0.32,
            "scanning patent text (chunked)")
    entity_agent = ChemicalEntityAgent(ai)
    chunks = _tagged_chunks(corpus, pages_per_chunk=4)
    mentions = _extract_entities_parallel(entity_agent, chunks)

    _notify(progress, "Chemical entities extracted", 0.40,
            f"{len(mentions)} candidate entities")

    # ------------------------------------------------------------------ #
    # 5. Compound validation
    # ------------------------------------------------------------------ #
    _notify(progress, "Validating compounds with chemistry APIs", 0.42,
            "OPSIN + PubChem + RDKit")
    compounds = _resolve_and_store_compounds(db, job, mentions, progress)

    # ------------------------------------------------------------------ #
    # 6. Reaction extraction
    # ------------------------------------------------------------------ #
    _notify(progress, "Extracting reactions", 0.58,
            "reading experimental examples")
    reactions = _extract_reactions(db, job, ai, corpus, sections, compounds, progress)

    # ------------------------------------------------------------------ #
    # 7. Reaction analysis (What Changed?)
    # ------------------------------------------------------------------ #
    _notify(progress, "Understanding reaction chemistry", 0.74,
            "RDKit structural diffs")
    _analyze_reactions(db, job, reactions, progress)

    # ------------------------------------------------------------------ #
    # 8. Manufacturing reconstruction
    # ------------------------------------------------------------------ #
    _notify(progress, "Reconstructing manufacturing process", 0.84,
            "stages + equipment")
    _extract_manufacturing(db, job, ai, corpus, sections, progress)

    # ------------------------------------------------------------------ #
    # 9. QC + summary
    # ------------------------------------------------------------------ #
    _notify(progress, "Quality control", 0.92, "hallucination checks")
    _summarize(job, extraction, ai)
    qc = run_qc(job)
    extra = dict(job.extra or {})
    extra.update({"qc_issues": [i.__dict__ for i in qc.issues], "qc_score": qc.score})
    job.extra = extra
    job.completed_at = datetime.now(timezone.utc)
    job.status = "complete"
    job.progress = 1.0
    job.current_stage = "Analysis complete"
    db.commit()

    try:
        build_graph(job)  # warm the flowchart cache
    except Exception as exc:  # noqa: BLE001
        logger.warning("flowchart warm failed: %s", exc)

    _notify(progress, "Analysis complete", 1.0,
            f"{len(compounds)} compounds, {len(reactions)} reactions")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tagged_chunks(corpus: list[dict], pages_per_chunk: int = 3) -> list[str]:
    """Group corpus chunks into page-tagged, LLM-friendly blocks."""
    by_page: dict[int, str] = {}
    for item in corpus:
        by_page.setdefault(item["page_no"], "")
        by_page[item["page_no"]] += "\n" + item["text"]
    pages = sorted(by_page)
    blocks: list[str] = []
    for i in range(0, len(pages), pages_per_chunk):
        group = pages[i : i + pages_per_chunk]
        parts = []
        for p in group:
            text = by_page[p].strip()
            if text:
                parts.append(f"[Patent page {p}]\n{text[:1000]}")
        if parts:
            blocks.append("\n\n".join(parts))
    return blocks


def _extract_entities_parallel(agent: ChemicalEntityAgent, chunks: list[str]) -> list:
    mentions = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(agent.extract_from_chunk, c): c for c in chunks}
        for fut in as_completed(futures):
            try:
                mentions.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("entity chunk failed: %s", exc)
    # Deduplicate by normalized name
    seen: set[str] = set()
    unique = []
    for m in mentions:
        key = m.name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique


def _resolve_and_store_compounds(db, job, mentions, progress) -> dict[str, Compound]:
    """Validate each unique entity via chemistry APIs + RDKit; store rows."""
    compounds: dict[str, Compound] = {}
    total = max(len(mentions), 1)
    for idx, m in enumerate(mentions):
        _notify(progress, "Validating compounds with chemistry APIs",
                0.42 + 0.15 * (idx / total), f"{m.name[:60]}")
        existing = compounds.get(m.name.strip().lower())
        if existing:
            if not existing.role or existing.role == "compound":
                existing.role = m.role
            continue

        rec = chemistry_service.resolve_compound(m.name)
        cid = f"C{len(compounds) + 1}"
        compound = Compound(
            job_id=job.id,
            cid=cid,
            name=m.name.strip(),
            role=m.role if m.role in {"reactant", "product", "intermediate", "compound"} else m.role,
            source_page=m.source_page,
            source_text=m.source_text,
            confidence=0.0,
            source_basis="patent",
            extra={},
        )
        if rec.resolved:
            compound.smiles = rec.canonical_smiles
            compound.iupac_name = rec.iupac_name or "Not specified in patent"
            compound.common_name = (rec.synonyms[0] if rec.synonyms else "Not specified in patent")
            compound.molecular_formula = rec.formula or "Not specified in patent"
            compound.molecular_weight = rec.molecular_weight
            compound.inchi = rec.inchi
            compound.inchikey = rec.inchikey
            compound.cas_number = rec.cas_number or "Not specified in patent"
            compound.confidence = rec.confidence
            compound.source_basis = "database"
            compound.extra = {
                "db_conflicts": rec.conflicts,
                "providers": rec.properties.get("providers_consulted", []),
            }
            # RDKit deep-verification + structure render
            info = analyze(rec.canonical_smiles or "")
            if info.valid:
                compound.structure_valid = True
                if not compound.molecular_formula or compound.molecular_formula == "Not specified in patent":
                    compound.molecular_formula = info.formula or compound.molecular_formula
                if compound.molecular_weight is None:
                    compound.molecular_weight = info.molecular_weight
                compound.extra["descriptors"] = {
                    "logp": info.logp, "tpsa": info.tpsa, "hbd": info.hbd,
                    "hba": info.hba, "rotatable_bonds": info.rotatable_bonds,
                    "aromatic_rings": info.aromatic_rings, "stereocenters": info.stereocenters,
                    "functional_groups": info.functional_groups,
                }
                compound.confidence = max(compound.confidence, 0.8)
                img_name = f"{job.id}_{cid}.png"
                if render_structure(compound.smiles, job_img_dir(job.id) / img_name):
                    compound.image_path = str(job_img_dir(job.id) / img_name)
            else:
                compound.confidence = 0.5
        db.add(compound)
        db.flush()
        compounds[m.name.strip().lower()] = compound

    db.commit()
    return compounds


def _extract_reactions(db, job, ai, corpus, sections, compounds, progress) -> list[Reaction]:
    # Target text: examples + process sections when present, else everything.
    focus: list[int] = []
    for s in sections:
        if s.kind in {"examples", "process"}:
            focus.extend(range(s.page_start, s.page_end + 1))
    if not focus:
        focus = list(range(1, job.page_count + 1))

    by_page = {c["page_no"]: c["text"] for c in corpus}
    blocks: list[str] = []
    for p in sorted(set(focus)):
        text = (by_page.get(p) or "").strip()
        if text:
            blocks.append(f"[Patent page {p}]\n{text[:1200]}")
    blocks = [b for b in blocks if b]

    agent = ReactionAnalysisAgent(ai)
    reactions: list[Reaction] = []
    r_count = 0
    for bi, block in enumerate(blocks):
        _notify(progress, "Extracting reactions", 0.58 + 0.14 * (bi / max(len(blocks), 1)),
                f"block {bi + 1}/{len(blocks)}")
        for extract in agent.extract_from_chunk(block):
            r_count += 1
            rxn = Reaction(
                job_id=job.id,
                rid=f"R{r_count}",
                name=extract.name,
                type=extract.type,
                reactants_text="; ".join(extract.reactants),
                products_text="; ".join(extract.products),
                reagents=extract.reagents,
                catalysts=extract.catalysts,
                solvents=extract.solvents,
                temperature=extract.temperature,
                pressure=extract.pressure,
                time=extract.time,
                atmosphere=extract.atmosphere,
                yield_pct=extract.yield_pct,
                workup=extract.workup,
                source_page=extract.source_page,
                source_text=extract.source_text,
                extra={"equipment": extract.equipment},
            )
            db.add(rxn)
            db.flush()
            _link_participants(db, job, rxn, extract, compounds)
            reactions.append(rxn)
    db.commit()
    return reactions


def _link_participants(db, job, rxn: Reaction, extract, compounds: dict[str, Compound]) -> None:
    """Match extracted reactant/product names to known compounds (or resolve new)."""
    from backend.models import compound_reaction

    for name, role in [(n, "reactant") for n in extract.reactants] + \
                       [(n, "product") for n in extract.products] + \
                       [(n, "intermediate") for n in _maybe_intermediates(extract)]:
        if not name or not name.strip():
            continue
        compound = _match_compound(name, compounds)
        if compound is None:
            compound = _resolve_new_compound(db, job, name, role, compounds)
        if compound is not None:
            db.execute(
                compound_reaction.insert().values(
                    reaction_id=rxn.id, compound_id=compound.id, role=role
                )
            )


def _maybe_intermediates(extract):
    return []


def _match_compound(name: str, compounds: dict[str, Compound]) -> Compound | None:
    key = name.strip().lower()
    if key in compounds:
        return compounds[key]
    # label-style references like "compound 3", "the product", "intermediate 2"
    import re

    m = re.search(r"(?:compound|intermediate|product)\s*([0-9]+|[ivxlcdm]+)", key)
    if m:
        return None  # resolution handled downstream by text matching
    # substring match fallback
    for k, c in compounds.items():
        if key and (key in c.name.lower() or c.name.lower() in key) and len(key) > 3:
            return c
    return None


def _resolve_new_compound(db, job, name: str, role: str, compounds: dict) -> Compound | None:
    rec = chemistry_service.resolve_compound(name)
    cid = f"C{len(compounds) + 1}"
    compound = Compound(
        job_id=job.id, cid=cid, name=name.strip(), role=role,
        source_basis="database", confidence=0.0, extra={},
    )
    if rec.resolved:
        compound.smiles = rec.canonical_smiles
        compound.iupac_name = rec.iupac_name or "Not specified in patent"
        compound.molecular_formula = rec.formula or "Not specified in patent"
        compound.molecular_weight = rec.molecular_weight
        compound.inchikey = rec.inchikey
        compound.cas_number = rec.cas_number or "Not specified in patent"
        compound.confidence = rec.confidence
        compound.extra = {"db_conflicts": rec.conflicts,
                          "providers": rec.properties.get("providers_consulted", [])}
        info = analyze(rec.canonical_smiles or "")
        if info.valid:
            compound.structure_valid = True
            compound.confidence = max(compound.confidence, 0.8)
            img_name = f"{job.id}_{cid}.png"
            if render_structure(compound.smiles, job_img_dir(job.id) / img_name):
                compound.image_path = str(job_img_dir(job.id) / img_name)
        db.add(compound)
        db.flush()
        compounds[name.strip().lower()] = compound
        return compound
    return None


def _analyze_reactions(db, job, reactions, progress) -> None:
    from backend.models import reaction_roles

    for i, rxn in enumerate(reactions):
        roles = reaction_roles(db, rxn.id)
        reactants = [p for p in rxn.participants if roles.get(p.id) == "reactant"]
        products = [p for p in rxn.participants if roles.get(p.id) == "product"]
        r_smiles = [c.smiles for c in reactants if c.smiles]
        p_smiles = [c.smiles for c in products if c.smiles]
        view = None
        if r_smiles and p_smiles:
            from backend.services.reaction_service import build_what_changed

            view = build_what_changed(r_smiles, p_smiles)
        if view is not None:
            rxn.what_changed = json.dumps(serialize(view))
            rxn.confidence = view.confidence
            rxn.reaction_smarts = "; ".join(view.reaction_types)
    db.commit()


def _extract_manufacturing(db, job, ai, corpus, sections, progress) -> None:
    focus: list[int] = []
    for s in sections:
        if s.kind in {"examples", "process", "description"}:
            focus.extend(range(s.page_start, s.page_end + 1))
    if not focus:
        focus = list(range(1, job.page_count + 1))

    by_page = {c["page_no"]: c["text"] for c in corpus}
    blocks: list[str] = []
    for p in sorted(set(focus))[:40]:
        text = (by_page.get(p) or "").strip()
        if text:
            blocks.append(f"[Patent page {p}]\n{text[:1500]}")

    agent = ManufacturingAgent(ai)
    merged: list[dict] = []
    raw, units, equip, scale_sum, notes = [], [], [], "Not specified in patent", "Not specified in patent"
    for bi, block in enumerate(blocks[:8]):
        _notify(progress, "Reconstructing manufacturing process", 0.84 + 0.04 * (bi / 8),
                f"page block {bi + 1}/{min(len(blocks), 8)}")
        result = agent.extract_from_chunk(block)
        for s in result.stages:
            s["order"] = len(merged) + 1
            merged.append(s)
        raw.extend(result.raw_materials)
        units.extend(result.process_units)
        equip.extend(result.equipment)
        if result.scale_summary != "Not specified in patent":
            scale_sum = result.scale_summary
        if result.notes != "Not specified in patent":
            notes = result.notes

    seen = set()
    for s in sorted(merged, key=lambda x: x.get("order", 0)):
        title = _strip_stage_prefix(s.get("title") or "")
        key = title.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        db.add(Stage(
            job_id=job.id, order_idx=s.get("order", len(seen)),
            title=title,
            purpose=_ns(s.get("purpose")), starting_material=_ns(s.get("starting_material")),
            reagents=_ns(s.get("reagents")), conditions=_ns(s.get("conditions")),
            reaction=_ns(s.get("reaction")), product=_ns(s.get("product")),
            what_changed=_ns(s.get("what_changed")), why_required=_ns(s.get("why_required")),
            chemistry=_ns(s.get("chemistry")), purification=_ns(s.get("purification")),
            yield_value=_ns(s.get("yield")), equipment=_ns(s.get("equipment")),
            patent_ref=_ns(s.get("patent_ref")), scale=_ns(s.get("scale")),
        ))

    extra = dict(job.extra or {})
    extra["scale_summary"] = scale_sum
    extra["manufacturing_notes"] = notes
    extra["process_units"] = sorted({u for u in units if u and u != "Not specified in patent"})
    job.extra = extra
    db.commit()


def _strip_stage_prefix(title: str) -> str:
    """Remove a leading "Stage N — " label the LLM prepends to titles."""
    import re

    return re.sub(r"^stage\s+\d+\s*[:\u2014-]\s*", "", title.strip(), flags=re.IGNORECASE).strip()


def _summarize(job, extraction, ai) -> None:
    header_text = "\n".join(p.text for p in extraction.pages[:6])[:6000]
    try:
        result = ai.chat_json(
            [system_message(SUMMARY_SYSTEM), user_message(SUMMARY_USER.format(text=header_text))],
            max_tokens=1200,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("summary extraction failed: %s", exc)
        return
    job.patent_title = (result.get("patent_title") or "").strip() or "Not specified in patent"
    job.patent_number = (result.get("patent_number") or "").strip() or "Not specified in patent"
    job.assignee = (result.get("assignee") or "").strip() or "Not specified in patent"
    job.inventors = (result.get("inventors") or "").strip() or "Not specified in patent"
    job.filing_date = (result.get("filing_date") or "").strip() or "Not specified in patent"


def _ns(value) -> str:
    v = (value or "").strip()
    return v if v else "Not specified in patent"


def job_img_dir(job_id: int) -> Path:
    from backend.config import settings

    d = settings.STRUCTURE_DIR / f"job_{job_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d
