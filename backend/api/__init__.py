"""FastAPI application factory and routers.

Endpoints:
  POST  /api/upload                          upload PDF, start analysis
  GET   /api/jobs/{job_id}/events            SSE progress stream
  GET   /api/jobs/{job_id}/summary           analysis summary
  GET   /api/jobs/{job_id}/compounds         detected compounds
  GET   /api/jobs/{job_id}/reactions         extracted reactions
  GET   /api/jobs/{job_id}/stages            stage-by-stage analysis
  GET   /api/jobs/{job_id}/flowchart         React Flow graph JSON
  GET   /api/jobs/{job_id}/manufacturing     manufacturing process view
  GET   /api/jobs/{job_id}/sources/{page}    patent source text for a page
  GET   /api/jobs/{job_id}/structures/{cid}.png  RDKit structure image
  GET   /api/jobs/{job_id}/report            download .docx report
  GET   /api/jobs                           list jobs
  GET   /api/health                         service health
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.agents.pipeline import run_pipeline
from backend.config import settings
from backend.database import get_db, init_db
from backend.models import Compound, Job, Page, Reaction, Stage
from backend.schemas import (
    AnalysisSummary,
    CompoundOut,
    FlowchartOut,
    FlowEdge,
    FlowNode,
    HealthOut,
    ManufacturingOut,
    ProgressEvent,
    ReactionOut,
    ReportOut,
    SectionOut,
    SourceOut,
    StageOut,
    UploadResponse,
)
from backend.services.flowchart_service import build_graph, to_react_flow
from backend.services.ocr_service import ocr_available
from backend.services.patent_service import detect_sections

logger = logging.getLogger("pca.api")

# In-memory progress queues: job_id -> queue of ProgressEvent dicts.
_progress_queues: dict[int, "queue.Queue[dict]"] = {}
_queues_lock = threading.Lock()


def _emit_progress(job_id: int, stage: str, pct: float, detail: str = "") -> None:
    with _queues_lock:
        q = _progress_queues.get(job_id)
    if q is not None:
        q.put({"stage": stage, "progress": pct, "detail": detail})


def _run_job_in_thread(job_id: int, pdf_path: Path) -> None:
    """Run the pipeline in a background thread with a fresh DB session."""
    from backend.database import SessionLocal

    try:
        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            if job is None:
                return
            run_pipeline(
                db,
                job,
                pdf_path,
                progress=lambda stage, pct, detail="": _emit_progress(job_id, stage, pct, detail),
            )
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s failed", job_id)
        with _queues_lock:
            q = _progress_queues.get(job_id)
        if q is not None:
            q.put({"stage": "failed", "progress": 1.0, "detail": str(exc)})
        try:
            db = SessionLocal()
            job = db.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error = str(exc)
                job.current_stage = "Analysis failed"
                db.commit()
            db.close()
        except Exception:  # noqa: BLE001
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings.ensure_dirs()
    init_db()
    _recover_orphaned_jobs()
    yield


def _recover_orphaned_jobs() -> None:
    """Mark jobs left 'processing'/'uploaded' by a previous server session.

    The pipeline runs in a daemon thread that dies with the process, so on a
    restart any in-flight job would otherwise be stuck forever.
    """
    from backend.database import SessionLocal
    from datetime import datetime, timezone

    with SessionLocal() as db:
        orphaned = (
            db.query(Job)
            .filter(Job.status.in_(["processing", "uploaded"]))
            .all()
        )
        for job in orphaned:
            job.status = "failed"
            job.error = "Analysis was interrupted by a server restart; upload the file again to retry."
            job.current_stage = "Analysis failed"
            job.completed_at = datetime.now(timezone.utc)
            logger.info("marked orphaned job %s (%s) as failed", job.id, job.status)
        if orphaned:
            db.commit()


def create_app() -> FastAPI:
    app = FastAPI(title="Patent Chemistry Analyzer API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthOut)
    def health():
        try:
            import rdkit  # noqa: F401

            rdkit_ok = True
        except Exception:  # noqa: BLE001
            rdkit_ok = False
        return HealthOut(provider=settings.active_ai_provider, tesseract=ocr_available(),
                         rdkit=rdkit_ok)

    @app.get("/api/jobs", response_model=list[AnalysisSummary])
    def list_jobs(db: Session = Depends(get_db)):
        jobs = db.query(Job).order_by(Job.created_at.desc()).limit(50).all()
        return [_summary(j) for j in jobs]

    @app.post("/api/upload", response_model=UploadResponse)
    async def upload(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are accepted.")

        settings.ensure_dirs()
        job = Job(filename=file.filename, original_filename=file.filename, status="uploaded")
        db.add(job)
        db.commit()
        db.refresh(job)

        dest = settings.UPLOAD_DIR / f"job_{job.id}.pdf"
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
            job.status = "failed"
            job.error = f"File too large (max {settings.MAX_UPLOAD_MB} MB)."
            db.commit()
            raise HTTPException(400, job.error)
        dest.write_bytes(content)
        job.file_size_bytes = len(content)

        # Validate it's actually a PDF and readable before starting.
        try:
            probe = Path(dest)
            import fitz

            doc = fitz.open(probe)
            page_count = doc.page_count
            doc.close()
            job.page_count = page_count
        except Exception:  # noqa: BLE001
            job.status = "failed"
            job.error = "The file could not be opened as a PDF."
            db.commit()
            raise HTTPException(400, job.error)

        with _queues_lock:
            _progress_queues[job.id] = queue.Queue()
        db.commit()

        thread = threading.Thread(
            target=_run_job_in_thread, args=(job.id, dest), daemon=True
        )
        thread.start()
        return UploadResponse(job_id=job.id, filename=file.filename)

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: int):
        """SSE stream of pipeline progress."""

        def gen():
            with _queues_lock:
                q = _progress_queues.setdefault(job_id, queue.Queue())
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    evt = q.get(timeout=1.0)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield f"event: progress\ndata: {evt}\n\n"
                if evt.get("stage") in {"complete", "failed"}:
                    return

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/summary", response_model=AnalysisSummary)
    def job_summary(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return _summary(job)

    @app.get("/api/jobs/{job_id}/compounds", response_model=list[CompoundOut])
    def compounds(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        out = []
        for c in job.compounds:
            co = CompoundOut.model_validate(c)
            co.image_url = f"/api/jobs/{job_id}/structures/{c.cid}.png"
            out.append(co)
        return out

    @app.get("/api/jobs/{job_id}/reactions", response_model=list[ReactionOut])
    def reactions(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return [_reaction_out(r) for r in sorted(job.reactions, key=lambda x: x.rid)]

    @app.get("/api/jobs/{job_id}/stages", response_model=list[StageOut])
    def stages(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        stages = sorted(job.stages, key=lambda x: x.order_idx)
        return [_stage_out(s) for s in stages]

    @app.get("/api/jobs/{job_id}/flowchart", response_model=FlowchartOut)
    def flowchart(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        nodes, edges = build_graph(job)
        from backend.services.flowchart_service import _hierarchical_layout
        layout = _hierarchical_layout(nodes, edges)
        node_list = []
        for n in nodes:
            pos = layout.get(n.id, {"x": 0, "y": 0})
            node_list.append(FlowNode(
                id=n.id, kind=n.kind, label=n.label,
                page=n.page, image_url=(
                    f"/api/jobs/{job_id}/structures/{n.cid}.png" if n.cid else None
                ),
                x=float(pos["x"]), y=float(pos["y"]),
            ))
        edge_list = []
        for e in edges:
            edge_list.append(FlowEdge(
                id=e.id, source=e.source, target=e.target, label=e.label,
            ))
        return FlowchartOut(job_id=job_id, nodes=node_list, edges=edge_list)

    @app.get("/api/jobs/{job_id}/manufacturing", response_model=ManufacturingOut)
    def manufacturing(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        stages_sorted = sorted(job.stages, key=lambda x: x.order_idx)
        extra = job.extra or {}
        return ManufacturingOut(
            job_id=job_id,
            scale_summary=extra.get("scale_summary", "Not specified in patent"),
            raw_materials=sorted({s.starting_material for s in stages_sorted
                                  if s.starting_material != "Not specified in patent"}),
            process_units=sorted({u for u in extra.get("process_units", [])
                                  if u and u != "Not specified in patent"}),
            equipment=sorted({e for s in stages_sorted if s.equipment != "Not specified in patent"
                              for e in s.equipment.replace(";", ",").split(",") if e.strip()}),
            stages=[_stage_out(s) for s in stages_sorted],
            notes=extra.get("manufacturing_notes", "Not specified in patent"),
        )

    @app.get("/api/jobs/{job_id}/sources/{page_no}", response_model=SourceOut)
    def source(job_id: int, page_no: int, db: Session = Depends(get_db)):
        page = (
            db.query(Page)
            .filter(Page.job_id == job_id, Page.page_no == page_no)
            .first()
        )
        if not page:
            raise HTTPException(404, "Page not found")
        return SourceOut(
            page_no=page.page_no,
            text=page.text or "Not specified in patent",
            image_url=(f"/api/jobs/{job_id}/page-image/{page_no}" if page.image_path else None),
        )

    @app.get("/api/jobs/{job_id}/page-image/{page_no}")
    def page_image(job_id: int, page_no: int, db: Session = Depends(get_db)):
        page = (
            db.query(Page)
            .filter(Page.job_id == job_id, Page.page_no == page_no)
            .first()
        )
        if not page or not page.image_path:
            raise HTTPException(404, "No image for this page")
        path = Path(page.image_path)
        if not path.exists():
            raise HTTPException(404, "Image file missing")
        return FileResponse(str(path), media_type="image/png")

    @app.get("/api/jobs/{job_id}/structures/{cid}.png")
    def structure_image(job_id: int, cid: str, db: Session = Depends(get_db)):
        compound = (
            db.query(Compound)
            .filter(Compound.job_id == job_id, Compound.cid == cid)
            .first()
        )
        if not compound or not compound.smiles:
            raise HTTPException(404, "Structure not available")
        from backend.agents.pipeline import job_img_dir

        path = Path(compound.image_path) if compound.image_path else None
        if not path or not path.exists():
            path = job_img_dir(job_id) / f"{job_id}_{cid}.png"
            from backend.services.structure_service import render_structure

            render_structure(compound.smiles, path)
        if not path.exists():
            raise HTTPException(404, "Structure image could not be rendered")
        return FileResponse(str(path), media_type="image/png")

    @app.get("/api/jobs/{job_id}/report", response_model=ReportOut)
    def report(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != "complete":
            raise HTTPException(400, "Analysis not complete yet.")
        from backend.services.report_service import generate_report

        out = generate_report(job, settings.REPORT_DIR)
        return ReportOut(filename=out.name, url=f"/api/jobs/{job_id}/report/download")

    @app.get("/api/jobs/{job_id}/report/download")
    def report_download(job_id: int, db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        safe = "".join(c if c.isalnum() or c in " .-_" else "_" for c in job.original_filename)
        path = settings.REPORT_DIR / f"{safe}-analysis.docx"
        if not path.exists():
            raise HTTPException(404, "Report not generated yet")
        return FileResponse(
            str(path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=path.name,
        )

    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


def _summary(job: Job) -> AnalysisSummary:
    return AnalysisSummary(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        page_count=job.page_count,
        filename=job.original_filename,
        patent_title=job.patent_title,
        patent_number=job.patent_number,
        assignee=job.assignee,
        inventors=job.inventors,
        filing_date=job.filing_date,
        sections=[SectionOut(id=s.id, kind=s.kind, page_start=s.page_start,
                             page_end=s.page_end, heading=s.heading) for s in job.sections],
        compound_count=len(job.compounds),
        reaction_count=len(job.reactions),
        stage_count=len(job.stages),
        confidence=(job.extra or {}).get("qc_score", 0.0),
        ocr_pages=sum(1 for p in job.pages if p.ocr_status in {"ocr", "mixed"}),
        database_conflicts=sum(1 for c in job.compounds if (c.extra or {}).get("db_conflicts")),
        error=job.error,
    )


def _stage_out(s: Stage) -> StageOut:
    return StageOut(
        id=s.id,
        order=s.order_idx,
        title=s.title,
        purpose=s.purpose or "Not specified in patent",
        starting_material=s.starting_material or "Not specified in patent",
        reagents=s.reagents or "Not specified in patent",
        conditions=s.conditions or "Not specified in patent",
        reaction=s.reaction or "Not specified in patent",
        product=s.product or "Not specified in patent",
        what_changed=s.what_changed or "Not specified in patent",
        why_required=s.why_required or "Not specified in patent",
        chemistry=s.chemistry or "Not specified in patent",
        purification=s.purification or "Not specified in patent",
        yield_value=s.yield_value or "Not specified in patent",
        equipment=s.equipment or "Not specified in patent",
        patent_ref=s.patent_ref or "Not specified in patent",
        scale=s.scale or "lab",
    )


def _reaction_out(r: Reaction) -> ReactionOut:
    from backend.models import reaction_roles
    from backend.database import SessionLocal
    from backend.schemas import ReactionParticipant

    with SessionLocal() as _db:
        roles = reaction_roles(_db, r.id)
    return ReactionOut(
        id=r.id,
        rid=r.rid,
        type=r.type or "Not specified in patent",
        name=r.name or "",
        reagents=r.reagents or "Not specified in patent",
        catalysts=r.catalysts or "Not specified in patent",
        solvents=r.solvents or "Not specified in patent",
        temperature=r.temperature or "Not specified in patent",
        pressure=r.pressure or "Not specified in patent",
        time=r.time or "Not specified in patent",
        atmosphere=r.atmosphere or "Not specified in patent",
        yield_pct=r.yield_pct or "Not specified in patent",
        workup=r.workup or "Not specified in patent",
        source_page=r.source_page,
        source_text=r.source_text or "",
        confidence=r.confidence or 0.0,
        what_changed=json.loads(r.what_changed) if r.what_changed else None,
        participants=[
            ReactionParticipant(
                cid=p.cid,
                name=p.name,
                role=roles.get(p.id, "reactant"),
                details="",
            )
            for p in r.participants
        ],
    )


app = create_app()
